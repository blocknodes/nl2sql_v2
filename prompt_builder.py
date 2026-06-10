"""
Prompt Builder v2 + BM25 Few-Shot Retriever
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


# ============== 型号提取与占位符 ==============

# 匹配常见型号模式
# 长型号：85U7S Pro+, KFR-35GW/KW1X-X1, BCD-650W80FZBAK
# 短系列：U7S, E8K, A7N（字母+数字+字母 or 字母+数字）
_MODEL_PATTERNS = [
    # 长型号：含连字符/斜杠/加号，或数字字母混合>=5字符
    re.compile(r'[A-Za-z0-9][A-Za-z0-9\-\+\.\/\#]{4,}(?:\s*(?:Pro\+?|Plus|Max|Ultra|Lite))?'),
    # 带数字前缀的型号：如 85U7S Pro+
    re.compile(r'\d{2,3}[A-Za-z][A-Za-z0-9\-\+\.\/]*(?:\s*(?:Pro\+?|Plus|Max|Ultra|Lite))?'),
    # 短系列代号：字母+数字+可选字母，如 U7S, E8K, A7N, E52Q
    re.compile(r'[A-Za-z]\d+[A-Za-z]*'),
]

# 品牌词不应被当作型号
_BRAND_WORDS = {"海信", "容声", "科龙", "维迪亚", "约克", "日立", "东芝", "ASKO", "TV", "Pro"}

# 参数值模式（不应被当作型号）：如 4GB, 64GB, 128MB, 4K, 8K, 120Hz, 1.5匹
_PARAM_VALUE_RE = re.compile(r'^\d+(\.\d+)?\s*(GB|MB|TB|KB|K|Hz|hz|匹|英寸|寸|W|kWh|rpm|dB|dBA|L|kg|mm|cm)(\+\d+(\.\d+)?\s*(GB|MB|TB|KB))?$', re.IGNORECASE)

MODEL_PLACEHOLDER = "__MODEL__"


def extract_model(query: str) -> Tuple[str, Optional[str]]:
    """
    从 query 中提取型号，返回 (替换后的query, 原始型号)。
    如果没有匹配到型号，返回原 query 和 None。
    优先取最长匹配。
    """
    candidates = []
    for pat in _MODEL_PATTERNS:
        for m in pat.finditer(query):
            token = m.group().strip()
            if token in _BRAND_WORDS or token.isdigit():
                continue
            # 过滤纯中文
            if not re.search(r'[A-Za-z0-9]', token):
                continue
            # 过滤参数值（如 4GB+64GB, 120Hz, 8K）
            if _PARAM_VALUE_RE.match(token):
                continue
            candidates.append((len(token), m.start(), token))

    if not candidates:
        return query, None

    # 取最长匹配，长度相同取最早出现的
    candidates.sort(key=lambda x: (-x[0], x[1]))
    model = candidates[0][2]
    replaced = query.replace(model, MODEL_PLACEHOLDER, 1)
    return replaced, model


def restore_model_in_dsl(dsl: Dict, model_value: Optional[str]) -> Dict:
    """将 DSL 中的 __MODEL__ 占位符还原为真实型号值，并确定正确的字段名。
    兜底：如果模型没有输出占位符但用了型号相关字段，也进行修正。
    """
    if model_value is None:
        return dsl

    _model_fields = {"productSeriesName", "promotionName", "salesModelName", "productModelName"}
    correct_field = _classify_model_field(model_value)
    found = False

    for f in dsl.get("filters") or []:
        if f.get("value") == MODEL_PLACEHOLDER or f.get("field") in _model_fields:
            f["value"] = model_value
            f["field"] = correct_field
            found = True
            break

    # 如果模型完全没输出型号相关 filter，补一个
    if not found:
        if dsl.get("filters") is None:
            dsl["filters"] = []
        dsl["filters"].append({"field": correct_field, "op": "=", "value": model_value})

    return dsl


def _classify_model_field(model_value: str) -> str:
    """根据型号格式判断应该用哪个字段。默认用 salesModelName。"""
    # 含中文 → promotionName（如"容声288S1""大薄荷E52Q"）
    if re.search(r'[\u4e00-\u9fff]', model_value):
        return "promotionName"
    # 默认用 salesModelName
    return "salesModelName"


# ============== BM25 简易实现（无外部依赖） ==============

def _tokenize(text: str) -> List[str]:
    """中文按字切，英文/数字按词切"""
    tokens = []
    for seg in re.findall(r'[\u4e00-\u9fff]|[A-Za-z0-9]+\.?[A-Za-z0-9]*', text):
        tokens.append(seg.lower())
    return tokens


class BM25:
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = [_tokenize(d) for d in corpus]
        self.n = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / max(self.n, 1)
        # df
        self.df: Dict[str, int] = {}
        for doc in self.docs:
            for t in set(doc):
                self.df[t] = self.df.get(t, 0) + 1

    def score(self, query: str, top_k: int = 3) -> List[int]:
        q_tokens = _tokenize(query)
        scores = []
        import math
        for i, doc in enumerate(self.docs):
            s = 0.0
            dl = len(doc)
            tf_map: Dict[str, int] = {}
            for t in doc:
                tf_map[t] = tf_map.get(t, 0) + 1
            for t in q_tokens:
                if t not in tf_map:
                    continue
                tf = tf_map[t]
                df = self.df.get(t, 0)
                idf = math.log((self.n - df + 0.5) / (df + 0.5) + 1)
                s += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            scores.append((s, i))
        scores.sort(reverse=True)
        return [idx for _, idx in scores[:top_k]]


# ============== Few-Shot Retriever ==============

class FewShotRetriever:
    def __init__(self, few_shots_path: Optional[str] = None):
        if few_shots_path is None:
            few_shots_path = str(Path(__file__).parent / "configs" / "few_shots.yaml")
        with open(few_shots_path, "r", encoding="utf-8") as f:
            self.examples: List[Dict] = yaml.safe_load(f) or []
        corpus = [ex["query"] for ex in self.examples]
        self.bm25 = BM25(corpus)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        indices = self.bm25.score(query, top_k)
        return [self.examples[i] for i in indices]


# ============== Prompt Builder ==============

# 字段字典（按类目）
BASE_FIELDS = """salesPriceYuan: 销售价格(元)
actualSalesDate: 实际销售时间
salesModelName: 销售型号名称，例如 YVOH260VAEMBQ，YVOH800VAEMCQ等
productModelName: 产品型号名称，例如 HKG-05DA/SG220XYBN#B，YUOH360VAEMCQ等
productPositionName: 产品定位名称，例如 高端，中端，低端，1档，2档，3档等
colorName: 颜色名称，例如 亚瑟银H200号，星云灰340号 等
netWeightKg: 净重/重量(kg)，例如 22，7.5
grossWeightKg: 毛重(kg)，例如 25，8.5
outerPackLengthMm: 外包装尺寸(长,mm)，例如 1430，560
outerPackWidthMm: 外包装尺寸(宽,mm)，例如 930，400
outerPackHeightMm: 外包装尺寸(高,mm)，例如 200，780
product_dimension_length_mm: 产品尺寸(长,mm)，例如 300，450
product_dimension_width_mm: 产品尺寸(宽,mm)，例如 200，350
product_dimension_height_mm: 产品尺寸(高,mm)，例如 100，150
domesticExportSaleName: 内销/外销名称，例如 内销，外销
productFamilyName: 产品家族名称，例如 M，U，Q等
productSeriesName: 产品系列名称，例如 日立U享系列，约克UD系列，XD3
salesModelLifeCycleStatusName: 销售型号生命周期状态名称，例如 退市准备，开发，上市，立项，作废等
salesAreaName: 销售区域名称，例如 中国，美国，德国等
promotionName: 推广名，例如 容声288S1，大薄荷E52Q等
productBigCategoryName: 大类名称，例如 显示类产品，清洁卫生器具等
productMidCategoryName: 中类名称，包含：电视、投影、显示器、冰箱、冷柜、洗衣机、烘干机、空调、灶具、热水器
productSmallCategoryName: 小类名称，例如平板电视、激光电视、波轮式洗衣机、滚筒式洗衣机等"""

CATEGORY_FIELDS = {
    "电视": """screenSizeInch: 屏幕尺寸(英寸)，例如80，110
resolution: 分辨率,字符串，例如2K、4K、5K、8K、FHD、HD、QHD、UHD、WQHD
resolutionHorizontalPixels: 分辨率-横向像素，整数，例如2160，4320；转化方式4k=2160
refreshRate: 刷新频率，例如 160Hz，120Hz
refreshRateHz: 刷新频率(Hz)，去除Hz单位后的整数，例如 160，120
ramCapacity: 运存RAM，例如 4GB，8GB
ramCapacityMb: 运存RAM(MB)，转为MB后的整数，例如 4096，65536
romCapacity: 存储ROM，例如 64GB，128GB
romCapacityMb: 存储ROM(MB)，转为MB后的整数，例如 65536
energyEfficiencyGrade: 能效等级，例如1级，2级，3级
productWidthWithoutBaseMm: 不含底座产品尺寸(宽,mm)，例如 1230，890
productHeightWithoutBaseMm: 不含底座产品尺寸(高,mm)，例如 710，520
productThicknessWithoutBaseMm: 不含底座产品尺寸(厚,mm)，例如 80，300
productWidthWithBaseMm: 含底座产品尺寸(宽,mm)，例如 1250，910
productHeightWithBaseMm: 含底座产品尺寸(高,mm)，例如 730，540
productThicknessWithBaseMm: 含底座产品尺寸(厚,mm)，例如 200，350""",
    "空调": """energyEfficiencyGrade: 能效等级，例如1级，2级，3级
product_dimension_width_mm: 产品尺寸(宽,mm)，例如 800，900
product_dimension_height_mm: 产品尺寸(高,mm)，例如 300，400
frequency_type: 变频/定频，例如 变频，定频
product_dimension_depth_mm: 产品尺寸(深,mm)
main_body_color: 外观主体颜色，例如紫砂咖，烟紫金，莫奈金等
indoor_unit_model: 内机产品型号，例如KFR-72L/QZ1-X1A(2X03)等
outdoor_unit_model: 外机产品型号，例如KFR-35W/H3V7X1(1X41)等
noise_dba: 噪音(dB(A))，例如 45，55
air_conditioner_type: 空调柜机或挂机，例如柜机、挂机
air_conditioner_horsepower: 空调匹数，例如1匹、1.5匹、2匹、3匹""",
    "冰箱": """product_dimension_width_mm: 产品尺寸(宽,mm)，例如 600，700
product_dimension_height_mm: 产品尺寸(高,mm)，例如 1800，2000
product_dimension_depth_mm: 产品尺寸(深,mm)
refrigerator_volume_l: 冷藏室容积(L)，例如 300，400
freezer_volume_l: 冷冻室容积(L)，例如 100，200
noise_dba: 噪音(dB(A))，例如 38，72
comprehensive_power_consumption_kwh_per_24h: 综合耗电量(kW·h/24h)，例如 1.7，2.5""",
    "冷柜": """energyEfficiencyGrade: 能效等级，例如1级，2级，3级
product_dimension_width_mm: 产品尺寸(宽,mm)，例如 800，900
product_dimension_height_mm: 产品尺寸(高,mm)，例如 850，950
frequency_type: 变频/定频方式，例如 变频、定频
product_dimension_depth_mm: 产品尺寸(深,mm)
door_color: 门体颜色，例如冰釉白401号，凯撒银291号等
cabinet_color: 箱体颜色，例如钛空金FL49-1，冷灰色FL50-1等
noise_dba: 噪音(dB(A))，例如 40，60
comprehensive_power_consumption_kwh_per_24h: 综合耗电量(kW·h/24h)，例如 1.2，2.0
total_volume_l: 总容积(L)，例如 300，400
temperature_zone: 温区，例如单温区、双温区、多温区""",
    "洗衣机": """energyEfficiencyGrade: 能效等级，例如1级，2级，3级
product_dimension_width_mm: 产品尺寸(宽,mm)，例如 600，700
product_dimension_height_mm: 产品尺寸(高,mm)，例如 850，950
key_press_method: 按键方式，例如 无、机械、触摸
nominal_drying_capacity_kg: 标称烘干容量(kg)，例如 5，6
nominal_dehydration_capacity_kg: 标称脱水容量(kg)，例如 7.5，8
nominal_washing_capacity_kg: 标称洗涤容量(kg)，例如 8，9
product_dimension_depth_mm: 产品尺寸(深,mm)
motor_type: 电机类型，例如BLDC电机、DDM电机、DD电机、串激电机、感应电机
rated_voltage_v: 额定电压(V)，例如 220，230，240
rated_frequency_hz: 额定频率(Hz)，例如60，50
drying_method: 烘干方式，例如 冷凝、无、热泵、直排
drainage_method: 排水方式，例如 上排水、下排水、水盒
appearance_color: 外观颜色，例如珠光白、黑色、白色等
washing_ratio: 洗净比，例如 1.2，1.5
display_type: 显示类型，例如 LED指示灯显示、内显、外显、None
noise_dba: 噪音(dB(A))，例如 50，60
refrigerant_type: 制冷剂种类，例如 R410A，R600a等
intelligent_dispensing: 智能投放，例如 单投、双投、无
max_dehydration_speed_rpm: 最高脱水转速(rpm)，例如 1200，1400
motor_fixed_or_inverter: 电机定变频，例如 定频、变频""",
    "烘干机": """product_dimension_width_mm: 产品尺寸(宽,mm)，例如 600，700
product_dimension_height_mm: 产品尺寸(高,mm)，例如 850，950
nominal_drying_capacity_kg: 标称烘干容量(kg)，例如 5，6
product_dimension_depth_mm: 产品尺寸(深,mm)
motor_type: 电机类型，例如 BLDC电机、DDM电机、DD电机、串激电机、感应电机
rated_voltage_v: 额定电压(V)，例如 220，230，240
rated_frequency_hz: 额定频率(Hz)，例如 60，50
drying_method: 烘干方式，例如 冷凝、无、热泵、直排
appearance_color: 外观颜色，例如 珠光白，黑色，白色等
noise_dba: 噪音(dB(A))，例如 50，60""",
    "投影": """screenSizeInch: 屏幕尺寸(英寸)，例如80，110
resolution: 分辨率，字符串，例如2K、4K、5K、8K、FHD、HD、QHD、UHD、WQHD
resolutionHorizontalPixels: 分辨率-横向像素，整数，例如2160，4320
refreshRate: 刷新频率，例如 160Hz，120Hz
refreshRateHz: 刷新频率(Hz)，去除Hz单位后的整数，例如 160，120
ramCapacity: 运存RAM，例如 4GB，8GB
ramCapacityMb: 运存RAM(MB)，转为MB后的整数
romCapacity: 存储ROM，例如 64GB，128GB
romCapacityMb: 存储ROM(MB)，转为MB后的整数
energyEfficiencyGrade: 能效等级，例如1级，2级，3级""",
    "显示器": """screenSizeInch: 屏幕尺寸(英寸)，例如24，27，32
resolution: 分辨率，字符串，例如2K、4K、5K、8K、FHD、HD、QHD、UHD、WQHD
resolutionHorizontalPixels: 分辨率-横向像素，整数，例如2160，4320
software_system: 软件系统，包含：Android, Chrome OS, Google, Non OS, VIDAA U8, 其他, 无操作系统
energyEfficiencyGrade: 能效等级，例如1级，2级，3级
product_dimension_width_no_base_mm: 不含底座产品尺寸(宽,mm)，例如 600，700
product_dimension_height_no_base_mm: 不含底座产品尺寸(高,mm)，例如 400，500
product_dimension_width_with_base_mm: 含底座产品尺寸(宽,mm)，例如 620，720
product_dimension_height_with_base_mm: 含底座产品尺寸(高,mm)，例如 420，520
product_dimension_height2_with_base_mm: 含底座产品尺寸(高2,mm)【仅高低可调底座填写】，例如 450，550
product_weight_no_base_kg: 不含底座产品重量(kg)，例如 3.5，5.0
product_dimension_length_no_base_mm: 不含底座产品尺寸(长,mm)，例如 150，200
product_dimension_length_with_base_mm: 含底座产品尺寸(长,mm)，例如 170，220""",
}


class PromptBuilderV2:
    def __init__(self, retriever: Optional[FewShotRetriever] = None):
        self.retriever = retriever or FewShotRetriever()

    def build(self, query: str, category: Optional[str] = None) -> str:
        # 动态召回 few-shot（用原始 query 召回）
        examples = self.retriever.retrieve(query, top_k=3)
        examples_str = self._format_examples(examples)

        # 字段字典
        fields_str = BASE_FIELDS
        if category and category in CATEGORY_FIELDS:
            fields_str += "\n" + CATEGORY_FIELDS[category]
        else:
            # 未知类目时全部展示
            for v in CATEGORY_FIELDS.values():
                fields_str += "\n" + v

        return f'''你是 NL2SQL 意图解析器。把用户自然语言转成下面格式的 JSON，不要输出任何其他文字。

# 输出格式
{{"category":"类目|null","brand":"中文品牌|null","intent":"list|price|spec|field","filters":[{{"field":"字段名","op":"运算符","value":"值"}}],"sort":{{"field":"字段名","dir":"asc|desc"}}|null,"limit":数字|null,"target_fields":["字段名"]|null}}

# intent 取值
- list: 问型号/有哪些/哪些产品/什么型号
- price: 问多少钱/价格/售价
- spec: 问参数/详情/规格/全部信息
- field: 问某个具体属性(如"几级能效""多重""什么颜色")

# brand 规则
直接输出中文品牌名(海信/容声/科龙/维迪亚/约克/日立/东芝/ASKO)。未提及品牌则null。

# category 取值
电视/空调/冰箱/冷柜/洗衣机/烘干机/投影/显示器 或 null

# op 取值
= > < >= <= != ~(约等于/左右)

# 可用 filter 字段
{fields_str}

# 规则
1. 价格区间(如3000-5000)用两个filter: op:">" 和 op:"<"
2. "xxx元左右"用 op:"~"
3. "带烘干"="烘干方式不为空"→ drying_method op:"!=" value:"NULL"
4. 匹数保留"匹"字(如"1.5匹")
5. "最贵/最便宜/最大"→ sort + limit:1
6. 刷新率数值比较(如>=120)用 refreshRateHz；等值(如=160Hz)用 refreshRate
7. 如果用户问的信息在上述字段字典中找不到(如订单、发货、支付、维修费、安装、生产批次等)，仍然尽力提取能识别的条件(品牌、类目、型号等)，target_fields留空
8. 用户只提到属性名而没有给具体数值时(如"电视毛重""空调噪音")，该属性放入target_fields，不要作为filter；只有给了具体值(如"毛重大于20kg")才放入filters
9. 如果输入中包含{MODEL_PLACEHOLDER}占位符，表示一个产品型号/系列名，直接在filter中使用field:"promotionName" op:"=" value:"{MODEL_PLACEHOLDER}"
10. "尺寸"指产品外观尺寸(宽高厚mm)，用target_fields查productWidthWithoutBaseMm/productHeightWithoutBaseMm/productThicknessWithoutBaseMm；"屏幕尺寸/多少寸"才用screenSizeInch

# 示例
{examples_str}
# 用户输入
{query}'''

    def _format_examples(self, examples: List[Dict]) -> str:
        lines = []
        for ex in examples:
            # 示例中的 query 也做型号占位替换，保持一致
            ex_query, _ = extract_model(ex["query"])
            dsl_str = json.dumps(ex["dsl"], ensure_ascii=False)
            lines.append(f'输入：{ex_query}\n输出：{dsl_str}')
        return "\n".join(lines)
