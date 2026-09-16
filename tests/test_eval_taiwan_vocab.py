from backend.pipeline.han_script import han_variant
from eval.taiwan_vocab import taiwan_pairs, to_taiwan


def test_conversion_uses_taiwan_vocabulary():
    assert to_taiwan("鼠标和内存") == "滑鼠和記憶體"


def test_lines_the_conversion_does_not_change_are_dropped():
    assert taiwan_pairs(["ABC 123", "计算机内存"]) == [("计算机内存", "計算機記憶體")]


def test_converted_text_reads_as_traditional_to_the_app_rule():
    _, converted = taiwan_pairs(["软体机器人使用硅胶"])[0]
    assert converted == "軟體機器人使用矽膠"
    assert han_variant(converted) == "zho_Hant"
    assert han_variant("软体机器人使用硅胶") == "zho_Hans"


def test_pairs_keep_the_simplified_source_line():
    assert taiwan_pairs(["如何制作软气动执行器"])[0][0] == "如何制作软气动执行器"
