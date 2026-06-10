"""
训练数据生成器（v2 DSL 格式）— 5000+ 条

通过模板排列组合 + 随机采样生成大量 query-DSL 对。
输出 JSONL，每行: {"id": N, "query": "...", "dsl": {...}}

用法:
  python scripts/gen_train_data.py --output data/train_v2.jsonl --count 5000
"""

import argparse
import json
import random
import itertools
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ============== 元数据 ==============

BRANDS = {
    "电视": ["海信", "维迪亚", "东芝"],
    "空调": ["海信", "科龙", "约克", "日立"],
    "冰箱": ["海信", "容声"],
    "冷柜": ["海信", "容声"],
    "洗衣机": ["海信", "容声", "ASKO"],
    "烘干机": ["海信", "ASKO"],
    "投影": ["海信", "维迪亚"],
    "显示器": ["海信"],
}

ALL_CATEGORIES = list(BRANDS.keys())

POSITIONS = ["高端", "中端", "低端", "1档", "2档", "3档"]

TV_MODELS = ["75L7Q-Pro", "65E8S", "100E8Q", "85E8S", "85U7Q", "65E7G", "55A6K", "75A7K", "100L5H",
             "85E8H", "65U7G", "55E7K", "75E3N", "85A7N", "65A8H", "55E8K-Pro", "100U8KL"]
AC_MODELS = ["KFR-35GW/KW1X-X1", "KFR-26G/LQ1U-X1", "KFR-72L/QZ1-X1A", "KFR-50GW/H5V7X1",
             "KFR-35W/H3V7X1", "KFR-26GW/K230D-A3"]
FRIDGE_MODELS = ["BCD-650W80FZBAK", "BCD-559WVK1FPG", "BCD-503WNK1FPG", "BCD-326WNK1EPC",
                 "BCD-252WD11NP", "BCD-430WD11FP"]
FREEZER_MODELS = ["BD/BC-715ZEL", "BD/BC-508ZEM", "BD/BC-305NEL", "BD/BC-202NET"]
WASHER_MODELS = ["WH130U9Q", "WD100H8", "WH130E7Q", "WD120H5", "WH100U9Q", "WD80T5"]
DRYER_MODELS = ["DH100D8", "DH90T7", "DH80D6", "B1406CYBA"]
PROJ_MODELS = ["P100", "PA80", "C1S", "L5H", "PX2-PRO"]
MONITOR_MODELS = ["32GX", "32X8N-PRO", "27X7N-PRO", "27G5F", "24N3G"]

MODELS_MAP = {
    "电视": TV_MODELS, "空调": AC_MODELS, "冰箱": FRIDGE_MODELS,
    "冷柜": FREEZER_MODELS, "洗衣机": WASHER_MODELS, "烘干机": DRYER_MODELS,
    "投影": PROJ_MODELS, "显示器": MONITOR_MODELS,
}

SCREEN_SIZES = [24, 27, 32, 43, 50, 55, 65, 75, 80, 85, 86, 98, 100, 110, 120]
RESOLUTIONS = ["4K", "8K", "FHD", "UHD", "2K", "QHD"]
REFRESH_RATES = ["60Hz", "90Hz", "120Hz", "144Hz", "160Hz", "165Hz", "240Hz"]
REFRESH_RATES_HZ = [60, 90, 120, 144, 160, 165, 240]
RAMS = ["2GB", "3GB", "4GB", "6GB", "8GB", "16GB"]
ROMS = ["16GB", "32GB", "64GB", "128GB", "256GB"]
ENERGY_GRADES = ["1级", "2级", "3级"]
HORSEPOWERS = ["1匹", "1.5匹", "2匹", "3匹", "5匹"]
FREQ_TYPES = ["变频", "定频"]
AC_TYPES = ["挂机", "柜机"]
WASH_KGS = [7, 8, 9, 10, 12, 13, 15]
DRY_KGS = [5, 6, 7, 8, 9, 10]
FRIDGE_VOL = [200, 250, 300, 350, 400, 450, 500, 550, 600]
FREEZER_VOL = [50, 80, 100, 120, 150, 180, 200]
TOTAL_VOL = [100, 150, 200, 300, 400, 500, 600, 700, 800]
TEMP_ZONES = ["单温区", "双温区", "多温区"]
MOTOR_TYPES = ["BLDC电机", "DDM电机", "DD电机", "串激电机", "感应电机"]
DRYING_METHODS = ["冷凝", "热泵", "直排"]
DRAINAGE_METHODS = ["上排水", "下排水"]
COLORS = ["白色", "黑色", "灰色", "银色", "金色", "蓝色", "珠光白", "星空灰", "莫奈金"]
NOISE_LEVELS = [35, 38, 40, 42, 45, 48, 50, 55, 60]
PRICES = [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 6000, 7000, 8000, 10000, 12000, 15000, 20000]

# ============== 工具 ==============

def dsl(category=None, brand=None, intent="list", filters=None, sort=None, limit=None, target_fields=None):
    d = {"category": category, "brand": brand, "intent": intent,
         "filters": filters or [], "sort": sort, "limit": limit}
    if target_fields:
        d["target_fields"] = target_fields
    return d


def f(field, op, value):
    return {"field": field, "op": op, "value": str(value)}


# ============== 生成器 ==============

class TrainDataGenerator:
    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        self.seen_queries = set()
        self.cases = []

    def _add(self, query: str, d: Dict):
        if query in self.seen_queries:
            return
        self.seen_queries.add(query)
        self.cases.append({"query": query, "dsl": d})

    def _brand_or_none(self, cat: str):
        """随机返回品牌或 None"""
        if self.rng.random() < 0.3:
            return None
        return self.rng.choice(BRANDS[cat])

    # ---------- 通用模板 ----------

    def gen_brand_category_list(self):
        """品牌+类目 查型号"""
        templates = [
            "{brand}{cat}有哪些型号", "{brand}{cat}型号", "{brand}的{cat}有哪些",
            "{brand}{cat}有什么", "查一下{brand}{cat}的型号", "{brand}有哪些{cat}",
            "{brand}{cat}都有哪些款", "{brand}家的{cat}型号",
        ]
        for cat in ALL_CATEGORIES:
            for brand in BRANDS[cat]:
                for tpl in templates:
                    q = tpl.format(brand=brand, cat=cat)
                    self._add(q, dsl(cat, brand))

    def gen_category_only(self):
        """纯类目"""
        templates = [
            "{cat}有哪些型号", "{cat}型号查询", "查一下{cat}",
            "有什么{cat}", "所有的{cat}", "{cat}列表",
        ]
        for cat in ALL_CATEGORIES:
            for tpl in templates:
                self._add(tpl.format(cat=cat), dsl(cat, None))

    def gen_price_compare(self):
        """价格比较"""
        for cat in ALL_CATEGORIES:
            for brand in BRANDS[cat] + [None]:
                for price in self.rng.sample(PRICES, min(6, len(PRICES))):
                    b_str = brand or ""
                    # 价格以上
                    self._add(f"{b_str}{price}元以上的{cat}",
                              dsl(cat, brand, filters=[f("salesPriceYuan", ">", price)]))
                    # 价格以下
                    self._add(f"{b_str}{price}元以下的{cat}",
                              dsl(cat, brand, filters=[f("salesPriceYuan", "<", price)]))
                    # 不超过
                    self._add(f"{b_str}不超过{price}元的{cat}",
                              dsl(cat, brand, filters=[f("salesPriceYuan", "<=", price)]))
                    # 左右
                    self._add(f"{b_str}{cat}{price}元左右",
                              dsl(cat, brand, filters=[f("salesPriceYuan", "~", price)]))

    def gen_price_range(self):
        """价格区间"""
        for cat in ALL_CATEGORIES:
            for brand in BRANDS[cat] + [None]:
                b_str = brand or ""
                for _ in range(4):
                    lo = self.rng.choice(PRICES[:10])
                    hi = self.rng.choice([p for p in PRICES if p > lo] or [lo + 1000])
                    self._add(f"{b_str}{lo}到{hi}元的{cat}",
                              dsl(cat, brand, filters=[f("salesPriceYuan", ">", lo), f("salesPriceYuan", "<", hi)]))
                    self._add(f"{b_str}{cat}价格在{lo}-{hi}之间",
                              dsl(cat, brand, filters=[f("salesPriceYuan", ">", lo), f("salesPriceYuan", "<", hi)]))

    def gen_sort_limit(self):
        """排序+limit"""
        patterns = [
            ("{b}{cat}最贵的", "desc", 1),
            ("{b}{cat}最便宜的", "asc", 1),
            ("{b}最贵的{cat}", "desc", 1),
            ("{b}最便宜的{cat}", "asc", 1),
            ("{b}{cat}最贵的3款", "desc", 3),
            ("{b}{cat}最贵的5款", "desc", 5),
            ("{b}{cat}最贵的10款", "desc", 10),
            ("{b}{cat}最便宜的3款", "asc", 3),
            ("{b}{cat}价格最高的", "desc", 1),
            ("{b}{cat}价格最低的", "asc", 1),
            ("{b}{cat}按价格从高到低", "desc", None),
            ("{b}{cat}按价格从低到高", "asc", None),
            ("{b}{cat}价格排行", "desc", None),
        ]
        for cat in ALL_CATEGORIES:
            for brand in BRANDS[cat] + [None]:
                b_str = brand or ""
                for tpl, dir_, lim in patterns:
                    q = tpl.format(b=b_str, cat=cat)
                    self._add(q, dsl(cat, brand, sort={"field": "salesPriceYuan", "dir": dir_}, limit=lim))

    def gen_position(self):
        """产品定位"""
        for cat in ALL_CATEGORIES:
            for brand in BRANDS[cat] + [None]:
                b_str = brand or ""
                for pos in POSITIONS[:3]:  # 高端/中端/低端
                    self._add(f"{b_str}{pos}{cat}",
                              dsl(cat, brand, filters=[f("productPositionName", "=", pos)]))
                    self._add(f"{b_str}{pos}{cat}有哪些",
                              dsl(cat, brand, filters=[f("productPositionName", "=", pos)]))
                    self._add(f"{b_str}{pos}{cat}最贵的",
                              dsl(cat, brand, filters=[f("productPositionName", "=", pos)],
                                  sort={"field": "salesPriceYuan", "dir": "desc"}, limit=1))

    def gen_model_query(self):
        """按型号查"""
        for cat, models in MODELS_MAP.items():
            for model in models:
                # 查价格
                self._add(f"{model}多少钱", dsl(None, None, intent="price", filters=[f("salesModelName", "=", model)]))
                self._add(f"{model}价格", dsl(None, None, intent="price", filters=[f("salesModelName", "=", model)]))
                self._add(f"{model}售价", dsl(None, None, intent="price", filters=[f("salesModelName", "=", model)]))
                # 查参数
                self._add(f"{model}参数", dsl(None, None, intent="spec", filters=[f("salesModelName", "=", model)]))
                self._add(f"{model}的规格", dsl(None, None, intent="spec", filters=[f("salesModelName", "=", model)]))
                self._add(f"{model}详细信息", dsl(None, None, intent="spec", filters=[f("salesModelName", "=", model)]))
                # 查型号是否存在
                self._add(f"{model}是什么{cat}", dsl(cat, None, filters=[f("salesModelName", "=", model)]))
                # 品牌+型号
                brand = self.rng.choice(BRANDS[cat])
                self._add(f"{brand}{model}{cat}", dsl(cat, brand, filters=[f("salesModelName", "=", model)]))

    # ---------- 电视特有 ----------

    def gen_tv_specific(self):
        cat = "电视"
        for brand in BRANDS[cat] + [None]:
            b_str = brand or ""
            # 屏幕尺寸
            for sz in SCREEN_SIZES:
                if sz < 40:
                    continue  # 电视不太可能小于40
                self._add(f"{b_str}{sz}英寸{cat}", dsl(cat, brand, filters=[f("screenSizeInch", "=", sz)]))
                self._add(f"{b_str}{sz}寸{cat}", dsl(cat, brand, filters=[f("screenSizeInch", "=", sz)]))
            # 屏幕尺寸 >=
            for sz in [65, 75, 85, 100]:
                self._add(f"{b_str}{sz}英寸以上的{cat}", dsl(cat, brand, filters=[f("screenSizeInch", ">=", sz)]))
                self._add(f"{b_str}{sz}寸以上{cat}", dsl(cat, brand, filters=[f("screenSizeInch", ">=", sz)]))
            # 分辨率
            for res in RESOLUTIONS:
                self._add(f"{b_str}{res}{cat}", dsl(cat, brand, filters=[f("resolution", "=", res)]))
                self._add(f"{b_str}{res}{cat}有哪些", dsl(cat, brand, filters=[f("resolution", "=", res)]))
            # 刷新率 >=
            for hz in [120, 144, 165]:
                self._add(f"{b_str}{hz}Hz以上的{cat}", dsl(cat, brand, filters=[f("refreshRateHz", ">=", hz)]))
            # 刷新率 =
            for rate in REFRESH_RATES:
                self._add(f"{b_str}{rate}{cat}", dsl(cat, brand, filters=[f("refreshRate", "=", rate)]))
            # 运存
            for ram in RAMS:
                self._add(f"{b_str}{ram}运存的{cat}", dsl(cat, brand, filters=[f("ramCapacity", "=", ram)]))
            # 存储
            for rom in ROMS:
                self._add(f"{b_str}{rom}存储的{cat}", dsl(cat, brand, filters=[f("romCapacity", "=", rom)]))

        # 复合条件
        for brand in BRANDS[cat]:
            for sz in [65, 75, 85, 100]:
                for res in ["4K", "8K"]:
                    self._add(f"{brand}{sz}英寸{res}{cat}",
                              dsl(cat, brand, filters=[f("screenSizeInch", "=", sz), f("resolution", "=", res)]))
            for sz in [75, 85, 100]:
                self._add(f"{brand}{sz}英寸以上最贵的{cat}",
                          dsl(cat, brand, filters=[f("screenSizeInch", ">=", sz)],
                              sort={"field": "salesPriceYuan", "dir": "desc"}, limit=1))
            for hz in [120, 144]:
                self._add(f"{brand}{hz}Hz以上4K{cat}",
                          dsl(cat, brand, filters=[f("refreshRateHz", ">=", hz), f("resolution", "=", "4K")]))

        # 查具体属性
        for brand in BRANDS[cat]:
            self._add(f"{brand}电视是几级能效",
                      dsl(cat, brand, intent="field", target_fields=["energyEfficiencyGrade"]))
            for sz in [65, 75, 85]:
                self._add(f"{brand}{sz}寸电视刷新率",
                          dsl(cat, brand, intent="field", filters=[f("screenSizeInch", "=", sz)], target_fields=["refreshRate"]))
                self._add(f"{brand}{sz}寸电视多重",
                          dsl(cat, brand, intent="field", filters=[f("screenSizeInch", "=", sz)], target_fields=["netWeightKg"]))

    # ---------- 空调特有 ----------

    def gen_ac_specific(self):
        cat = "空调"
        for brand in BRANDS[cat] + [None]:
            b_str = brand or ""
            # 匹数
            for hp in HORSEPOWERS:
                self._add(f"{b_str}{hp}{cat}", dsl(cat, brand, filters=[f("air_conditioner_horsepower", "=", hp)]))
            # 变频/定频
            for ft in FREQ_TYPES:
                self._add(f"{b_str}{ft}{cat}", dsl(cat, brand, filters=[f("frequency_type", "=", ft)]))
            # 挂机/柜机
            for at in AC_TYPES:
                self._add(f"{b_str}{at}{cat}", dsl(cat, brand, filters=[f("air_conditioner_type", "=", at)]))
            # 能效
            for eg in ENERGY_GRADES:
                self._add(f"{b_str}{eg}能效{cat}", dsl(cat, brand, filters=[f("energyEfficiencyGrade", "=", eg)]))
            # 噪音
            for n in [45, 50, 55]:
                self._add(f"{b_str}噪音{n}分贝以下的{cat}", dsl(cat, brand, filters=[f("noise_dba", "<", n)]))

        # 复合条件
        for brand in BRANDS[cat]:
            for hp in HORSEPOWERS[:4]:
                for ft in FREQ_TYPES:
                    self._add(f"{brand}{hp}{ft}{cat}",
                              dsl(cat, brand, filters=[f("air_conditioner_horsepower", "=", hp), f("frequency_type", "=", ft)]))
                for at in AC_TYPES:
                    self._add(f"{brand}{hp}{at}{cat}",
                              dsl(cat, brand, filters=[f("air_conditioner_horsepower", "=", hp), f("air_conditioner_type", "=", at)]))
            for hp in ["1.5匹", "2匹", "3匹"]:
                for ft in FREQ_TYPES:
                    for at in AC_TYPES:
                        self._add(f"{brand}{hp}{ft}{at}{cat}",
                                  dsl(cat, brand, filters=[
                                      f("air_conditioner_horsepower", "=", hp),
                                      f("frequency_type", "=", ft),
                                      f("air_conditioner_type", "=", at)]))

        # 查属性
        for brand in BRANDS[cat]:
            self._add(f"{brand}空调是几级能效",
                      dsl(cat, brand, intent="field", target_fields=["energyEfficiencyGrade"]))

    # ---------- 冰箱特有 ----------

    def gen_fridge_specific(self):
        cat = "冰箱"
        for brand in BRANDS[cat] + [None]:
            b_str = brand or ""
            # 冷藏容积
            for vol in FRIDGE_VOL:
                self._add(f"{b_str}冷藏室{vol}升以上的{cat}",
                          dsl(cat, brand, filters=[f("refrigerator_volume_l", ">", vol)]))
                self._add(f"{b_str}冷藏{vol}L以上{cat}",
                          dsl(cat, brand, filters=[f("refrigerator_volume_l", ">", vol)]))
            # 冷冻容积
            for vol in FREEZER_VOL:
                self._add(f"{b_str}冷冻室{vol}升以上的{cat}",
                          dsl(cat, brand, filters=[f("freezer_volume_l", ">", vol)]))
            # 噪音
            for n in [38, 40, 42, 45]:
                self._add(f"{b_str}噪音{n}分贝以下的{cat}",
                          dsl(cat, brand, filters=[f("noise_dba", "<", n)]))
            # 耗电量
            for kwh in ["1.0", "1.5", "2.0"]:
                self._add(f"{b_str}日耗电{kwh}度以下的{cat}",
                          dsl(cat, brand, filters=[f("comprehensive_power_consumption_kwh_per_24h", "<", kwh)]))

        # 复合
        for brand in BRANDS[cat]:
            for fv in [300, 400, 500]:
                for zv in [100, 150]:
                    self._add(f"{brand}冷藏{fv}升冷冻{zv}升以上的{cat}",
                              dsl(cat, brand, filters=[f("refrigerator_volume_l", ">", fv), f("freezer_volume_l", ">", zv)]))

        # 查属性
        for brand in BRANDS[cat]:
            self._add(f"{brand}冰箱耗电量",
                      dsl(cat, brand, intent="field", target_fields=["comprehensive_power_consumption_kwh_per_24h"]))
            self._add(f"{brand}冰箱噪音多大",
                      dsl(cat, brand, intent="field", target_fields=["noise_dba"]))

    # ---------- 冷柜特有 ----------

    def gen_freezer_specific(self):
        cat = "冷柜"
        for brand in BRANDS[cat] + [None]:
            b_str = brand or ""
            for vol in TOTAL_VOL:
                self._add(f"{b_str}{vol}升以上的{cat}",
                          dsl(cat, brand, filters=[f("total_volume_l", ">", vol)]))
            for tz in TEMP_ZONES:
                self._add(f"{b_str}{tz}{cat}", dsl(cat, brand, filters=[f("temperature_zone", "=", tz)]))
            for eg in ENERGY_GRADES:
                self._add(f"{b_str}{eg}能效{cat}", dsl(cat, brand, filters=[f("energyEfficiencyGrade", "=", eg)]))
            for ft in FREQ_TYPES:
                self._add(f"{b_str}{ft}{cat}", dsl(cat, brand, filters=[f("frequency_type", "=", ft)]))

        # 复合
        for brand in BRANDS[cat]:
            for vol in [300, 500]:
                for tz in TEMP_ZONES:
                    self._add(f"{brand}{vol}升以上{tz}{cat}",
                              dsl(cat, brand, filters=[f("total_volume_l", ">", vol), f("temperature_zone", "=", tz)]))

    # ---------- 洗衣机特有 ----------

    def gen_washer_specific(self):
        cat = "洗衣机"
        for brand in BRANDS[cat] + [None]:
            b_str = brand or ""
            # 洗涤容量
            for kg in WASH_KGS:
                self._add(f"{b_str}{kg}公斤{cat}", dsl(cat, brand, filters=[f("nominal_washing_capacity_kg", "=", kg)]))
                self._add(f"{b_str}{kg}kg{cat}", dsl(cat, brand, filters=[f("nominal_washing_capacity_kg", "=", kg)]))
            # 带烘干
            self._add(f"{b_str}带烘干的{cat}", dsl(cat, brand, filters=[f("drying_method", "!=", "NULL")]))
            # 电机类型
            for mt in MOTOR_TYPES:
                self._add(f"{b_str}{mt}{cat}", dsl(cat, brand, filters=[f("motor_type", "=", mt)]))
            # 排水方式
            for dm in DRAINAGE_METHODS:
                self._add(f"{b_str}{dm}{cat}", dsl(cat, brand, filters=[f("drainage_method", "=", dm)]))
            # 变频
            self._add(f"{b_str}变频{cat}", dsl(cat, brand, filters=[f("motor_fixed_or_inverter", "=", "变频")]))
            # 颜色
            for c in ["白色", "灰色", "银色"]:
                self._add(f"{b_str}{c}{cat}", dsl(cat, brand, filters=[f("appearance_color", "=", c)]))
            # 能效
            for eg in ENERGY_GRADES:
                self._add(f"{b_str}{eg}能效{cat}", dsl(cat, brand, filters=[f("energyEfficiencyGrade", "=", eg)]))

        # 复合：容量+烘干
        for brand in BRANDS[cat]:
            for kg in [10, 12, 13]:
                self._add(f"{brand}{kg}公斤带烘干的{cat}",
                          dsl(cat, brand, filters=[f("nominal_washing_capacity_kg", "=", kg), f("drying_method", "!=", "NULL")]))
                self._add(f"{brand}{kg}公斤带烘干{cat}价格从低到高",
                          dsl(cat, brand, filters=[f("nominal_washing_capacity_kg", "=", kg), f("drying_method", "!=", "NULL")],
                              sort={"field": "salesPriceYuan", "dir": "asc"}))
            for kg in WASH_KGS:
                self._add(f"{brand}{kg}公斤变频{cat}",
                          dsl(cat, brand, filters=[f("nominal_washing_capacity_kg", "=", kg), f("motor_fixed_or_inverter", "=", "变频")]))

    # ---------- 烘干机特有 ----------

    def gen_dryer_specific(self):
        cat = "烘干机"
        for brand in BRANDS[cat] + [None]:
            b_str = brand or ""
            for kg in DRY_KGS:
                self._add(f"{b_str}{kg}公斤{cat}", dsl(cat, brand, filters=[f("nominal_drying_capacity_kg", "=", kg)]))
            for dm in DRYING_METHODS:
                self._add(f"{b_str}{dm}{cat}", dsl(cat, brand, filters=[f("drying_method", "=", dm)]))

        # 复合
        for brand in BRANDS[cat]:
            for kg in [8, 9, 10]:
                for dm in DRYING_METHODS:
                    self._add(f"{brand}{kg}公斤{dm}{cat}",
                              dsl(cat, brand, filters=[f("nominal_drying_capacity_kg", "=", kg), f("drying_method", "=", dm)]))

    # ---------- 投影特有 ----------

    def gen_projector_specific(self):
        cat = "投影"
        for brand in BRANDS[cat] + [None]:
            b_str = brand or ""
            for res in RESOLUTIONS[:4]:
                self._add(f"{b_str}{res}投影仪", dsl(cat, brand, filters=[f("resolution", "=", res)]))
                self._add(f"{b_str}{res}投影", dsl(cat, brand, filters=[f("resolution", "=", res)]))
            for eg in ENERGY_GRADES:
                self._add(f"{b_str}{eg}能效投影", dsl(cat, brand, filters=[f("energyEfficiencyGrade", "=", eg)]))

        for brand in BRANDS[cat]:
            self._add(f"{brand}4K投影仪最贵的3款",
                      dsl(cat, brand, filters=[f("resolution", "=", "4K")],
                          sort={"field": "salesPriceYuan", "dir": "desc"}, limit=3))

    # ---------- 显示器特有 ----------

    def gen_monitor_specific(self):
        cat = "显示器"
        for brand in BRANDS[cat] + [None]:
            b_str = brand or ""
            for sz in [24, 27, 32]:
                self._add(f"{b_str}{sz}英寸{cat}", dsl(cat, brand, filters=[f("screenSizeInch", "=", sz)]))
                for res in ["2K", "4K", "FHD"]:
                    self._add(f"{b_str}{sz}英寸{res}{cat}",
                              dsl(cat, brand, filters=[f("screenSizeInch", "=", sz), f("resolution", "=", res)]))
                for rate in ["120Hz", "144Hz", "160Hz", "240Hz"]:
                    self._add(f"{b_str}{sz}英寸{rate}{cat}",
                              dsl(cat, brand, filters=[f("screenSizeInch", "=", sz), f("refreshRate", "=", rate)]))
            for res in RESOLUTIONS[:4]:
                self._add(f"{b_str}{res}{cat}", dsl(cat, brand, filters=[f("resolution", "=", res)]))

        # 复合 + 查属性
        for brand in BRANDS[cat]:
            for sz in [27, 32]:
                for rate in ["144Hz", "160Hz"]:
                    self._add(f"{brand}{sz}英寸{rate}{cat}是几级能效",
                              dsl(cat, brand, intent="field",
                                  filters=[f("screenSizeInch", "=", sz), f("refreshRate", "=", rate)],
                                  target_fields=["energyEfficiencyGrade"]))

    # ---------- 额外变体：查询表述多样性 ----------

    def gen_query_variants(self):
        """针对已有 case 用不同表述再生成一批"""
        variant_templates = {
            "list": [
                "帮我查一下{b}{cat}{cond_desc}", "我想看看{b}{cat}{cond_desc}",
                "{b}{cat}{cond_desc}有几款", "有没有{b}{cond_desc}的{cat}",
                "{b}{cond_desc}{cat}推荐", "推荐{b}{cond_desc}的{cat}",
            ],
            "price": [
                "{model}现在多少钱", "{model}的价格是多少", "{model}卖多少",
                "查一下{model}价格", "{model}的售价",
            ],
            "spec": [
                "{model}的详细参数", "{model}配置", "查一下{model}的规格",
                "{model}的所有信息",
            ],
        }

        # list 变体
        combos = [
            ("电视", "海信", "75英寸", [f("screenSizeInch", "=", "75")]),
            ("电视", "维迪亚", "4K", [f("resolution", "=", "4K")]),
            ("空调", "科龙", "1.5匹变频", [f("air_conditioner_horsepower", "=", "1.5匹"), f("frequency_type", "=", "变频")]),
            ("空调", "海信", "2匹柜机", [f("air_conditioner_horsepower", "=", "2匹"), f("air_conditioner_type", "=", "柜机")]),
            ("冰箱", "容声", "冷藏400升以上", [f("refrigerator_volume_l", ">", "400")]),
            ("洗衣机", "海信", "10公斤", [f("nominal_washing_capacity_kg", "=", "10")]),
            ("洗衣机", "海信", "带烘干", [f("drying_method", "!=", "NULL")]),
            ("烘干机", "ASKO", "热泵", [f("drying_method", "=", "热泵")]),
            ("冷柜", "容声", "500升以上", [f("total_volume_l", ">", "500")]),
            ("显示器", "海信", "32英寸4K", [f("screenSizeInch", "=", "32"), f("resolution", "=", "4K")]),
        ]
        for cat, brand, cond_desc, filters in combos:
            for tpl in variant_templates["list"]:
                q = tpl.format(b=brand, cat=cat, cond_desc=cond_desc)
                self._add(q, dsl(cat, brand, filters=filters))

        # price/spec 变体
        for model in self.rng.sample(TV_MODELS + AC_MODELS + WASHER_MODELS, 15):
            for tpl in variant_templates["price"]:
                self._add(tpl.format(model=model), dsl(None, None, intent="price", filters=[f("salesModelName", "=", model)]))
            for tpl in variant_templates["spec"]:
                self._add(tpl.format(model=model), dsl(None, None, intent="spec", filters=[f("salesModelName", "=", model)]))

    # ---------- 额外：含排序 + position 组合 ----------

    def gen_position_sort_combos(self):
        """定位 + 排序组合"""
        for cat in ALL_CATEGORIES:
            for brand in BRANDS[cat]:
                for pos in ["高端", "中端"]:
                    for lim in [3, 5]:
                        self._add(f"{brand}{pos}{cat}最贵的{lim}款",
                                  dsl(cat, brand,
                                      filters=[f("productPositionName", "=", pos)],
                                      sort={"field": "salesPriceYuan", "dir": "desc"}, limit=lim))
                        self._add(f"{brand}{pos}{cat}最便宜的{lim}款",
                                  dsl(cat, brand,
                                      filters=[f("productPositionName", "=", pos)],
                                      sort={"field": "salesPriceYuan", "dir": "asc"}, limit=lim))

    # ---------- 额外：价格+其他条件组合 ----------

    def gen_price_plus_filter(self):
        """价格条件 + 其他筛选"""
        combos = [
            ("电视", "海信", [("screenSizeInch", "=", "75")], "75英寸"),
            ("电视", "维迪亚", [("resolution", "=", "4K")], "4K"),
            ("电视", "海信", [("refreshRateHz", ">=", "120")], "120Hz以上"),
            ("空调", "科龙", [("air_conditioner_horsepower", "=", "1.5匹")], "1.5匹"),
            ("空调", "海信", [("frequency_type", "=", "变频")], "变频"),
            ("冰箱", "容声", [("refrigerator_volume_l", ">", "350")], "冷藏350升以上"),
            ("洗衣机", "海信", [("nominal_washing_capacity_kg", "=", "10")], "10公斤"),
        ]
        for cat, brand, extra_filters, desc in combos:
            for price in [3000, 5000, 8000]:
                filters = [f(ff[0], ff[1], ff[2]) for ff in extra_filters]
                # 价格以下
                self._add(f"{brand}{desc}{price}元以下的{cat}",
                          dsl(cat, brand, filters=filters + [f("salesPriceYuan", "<", price)]))
                # 价格以上
                self._add(f"{brand}{desc}{price}元以上的{cat}",
                          dsl(cat, brand, filters=filters + [f("salesPriceYuan", ">", price)]))
                # 左右
                self._add(f"{brand}{desc}{cat}{price}元左右",
                          dsl(cat, brand, filters=filters + [f("salesPriceYuan", "~", price)]))

    # ---------- 主生成 ----------

    def generate(self, target_count: int = 5000) -> List[Dict]:
        # 基础模板
        self.gen_brand_category_list()
        self.gen_category_only()
        self.gen_price_compare()
        self.gen_price_range()
        self.gen_sort_limit()
        self.gen_position()
        self.gen_model_query()

        # 类目特有
        self.gen_tv_specific()
        self.gen_ac_specific()
        self.gen_fridge_specific()
        self.gen_freezer_specific()
        self.gen_washer_specific()
        self.gen_dryer_specific()
        self.gen_projector_specific()
        self.gen_monitor_specific()

        # 组合扩展
        self.gen_query_variants()
        self.gen_position_sort_combos()
        self.gen_price_plus_filter()

        # 如果不够 target_count，通过随机扰动补充
        if len(self.cases) < target_count:
            self._augment(target_count - len(self.cases))

        # 截断到 target_count
        if len(self.cases) > target_count:
            self.rng.shuffle(self.cases)
            self.cases = self.cases[:target_count]

        # 分配 id
        self.rng.shuffle(self.cases)
        for i, c in enumerate(self.cases, 1):
            c["id"] = i

        return self.cases

    def _augment(self, needed: int):
        """通过随机组合补充数据"""
        prefixes = ["查一下", "帮我看看", "我想了解", "请问", ""]
        suffixes = ["", "有哪些", "推荐一下", "都有什么"]

        for _ in range(needed * 3):  # 多生成一些去重
            if len(self.cases) >= len(self.seen_queries) + needed:
                break
            cat = self.rng.choice(ALL_CATEGORIES)
            brand = self._brand_or_none(cat)
            b_str = brand or ""
            prefix = self.rng.choice(prefixes)
            suffix = self.rng.choice(suffixes)

            # 随机选特征
            filters = []
            desc_parts = []
            r = self.rng.random()

            if cat == "电视" and r < 0.5:
                sz = self.rng.choice(SCREEN_SIZES[4:])  # >= 50
                filters.append(f("screenSizeInch", "=", sz))
                desc_parts.append(f"{sz}英寸")
                if self.rng.random() < 0.5:
                    res = self.rng.choice(["4K", "8K"])
                    filters.append(f("resolution", "=", res))
                    desc_parts.append(res)
            elif cat == "空调" and r < 0.5:
                hp = self.rng.choice(HORSEPOWERS[:4])
                filters.append(f("air_conditioner_horsepower", "=", hp))
                desc_parts.append(hp)
                if self.rng.random() < 0.5:
                    ft = self.rng.choice(FREQ_TYPES)
                    filters.append(f("frequency_type", "=", ft))
                    desc_parts.append(ft)
            elif cat == "洗衣机" and r < 0.5:
                kg = self.rng.choice(WASH_KGS)
                filters.append(f("nominal_washing_capacity_kg", "=", kg))
                desc_parts.append(f"{kg}公斤")
            elif cat == "冰箱" and r < 0.5:
                vol = self.rng.choice(FRIDGE_VOL)
                filters.append(f("refrigerator_volume_l", ">", vol))
                desc_parts.append(f"冷藏{vol}升以上")

            # 可能加价格
            if self.rng.random() < 0.3:
                price = self.rng.choice(PRICES)
                op = self.rng.choice(["<", ">", "~"])
                filters.append(f("salesPriceYuan", op, price))
                if op == "<":
                    desc_parts.append(f"{price}元以下")
                elif op == ">":
                    desc_parts.append(f"{price}元以上")
                else:
                    desc_parts.append(f"{price}元左右")

            # 可能加排序
            sort = None
            limit = None
            if self.rng.random() < 0.2:
                dir_ = self.rng.choice(["desc", "asc"])
                sort = {"field": "salesPriceYuan", "dir": dir_}
                limit = self.rng.choice([1, 3, 5, None])
                if dir_ == "desc" and limit:
                    desc_parts.append(f"最贵的{limit}款")
                elif dir_ == "asc" and limit:
                    desc_parts.append(f"最便宜的{limit}款")

            desc = "".join(desc_parts)
            q = f"{prefix}{b_str}{desc}{cat}{suffix}".strip()
            self._add(q, dsl(cat, brand, filters=filters, sort=sort, limit=limit))


# ============== 主入口 ==============

def main():
    p = argparse.ArgumentParser(description="生成 v2 训练数据（5000条）")
    p.add_argument("--output", default="data/train_v2.jsonl")
    p.add_argument("--count", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    gen = TrainDataGenerator(seed=args.seed)
    cases = gen.generate(target_count=args.count)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fp:
        for c in cases:
            fp.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"[gen] {len(cases)} cases → {out_path}")


if __name__ == "__main__":
    main()
