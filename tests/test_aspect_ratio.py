from pipeline.checks import aspect_ratio


def find(issues, code):
    return [(i.image, i.line) for i in issues if i.code == code]


def test_normal_aspect_ratio_and_balanced_classes_are_silent(make_ds, cfg):
    ds = make_ds(images={"a": 1}, labels={"a": ["0 0.5 0.5 0.2 0.2", "1 0.3 0.3 0.1 0.2"]}, size=(640, 480))
    assert aspect_ratio.run(ds, cfg) == []


def test_extreme_wide_and_tall_boxes_are_flagged(make_ds, cfg):
    # dòng 1: rộng 0.5*640=320px, cao 0.05*480=24px -> ratio 13.33 > 6.0
    # dòng 2: bình thường 0.2*640=128px, 0.2*480=96px -> ratio 1.33
    # dòng 3: rộng 0.03*640=19.2px, cao 0.4*480=192px -> ratio 10.0 > 6.0
    ds = make_ds(
        images={"a": 1},
        labels={"a": ["0 0.5 0.5 0.5 0.05", "0 0.5 0.5 0.2 0.2", "1 0.5 0.5 0.03 0.4"]},
        size=(640, 480),
    )
    iss = aspect_ratio.run(ds, cfg)
    assert find(iss, "BOX_EXTREME_ASPECT") == [("a", 1), ("a", 3)]
    assert all(i.severity == "warn" and i.extra["aspect_ratio"] > 6.0 for i in iss if i.code == "BOX_EXTREME_ASPECT")


def test_class_imbalance_detected(make_ds, cfg):
    # lớp 1 có 7 box, lớp 2 chỉ có 1 box -> tỉ số 7.0 > 5.0
    lines = ["1 0.5 0.5 0.2 0.2"] * 7 + ["2 0.5 0.5 0.2 0.2"]
    ds = make_ds(images={"a": 1}, labels={"a": lines}, size=(640, 480))
    iss = aspect_ratio.run(ds, cfg)
    imb = [i for i in iss if i.code == "CLASS_IMBALANCE"]
    assert len(imb) == 1 and imb[0].image == "<dataset>" and imb[0].extra["ratio"] == 7.0


def test_custom_thresholds_from_config(make_ds, cfg):
    cfg["aspect_ratio"] = {"max_aspect_ratio": 20.0, "max_imbalance_ratio": 10.0}
    lines = ["1 0.5 0.5 0.5 0.05"] * 7 + ["2 0.5 0.5 0.2 0.2"]
    ds = make_ds(images={"a": 1}, labels={"a": lines}, size=(640, 480))
    assert aspect_ratio.run(ds, cfg) == []


def test_broken_lines_and_blank_lines_do_not_crash(make_ds, cfg):
    ds = make_ds(
        images={"a": 1},
        labels={"a": ["garbage", "", "93 0.5 0.5 0.5 0.01", "0 0.5 0.5 0 0.2", "1 0.5 0.5 0.5 0.04"]},
        size=(640, 480),
    )
    iss = aspect_ratio.run(ds, cfg)
    assert find(iss, "BOX_EXTREME_ASPECT") == [("a", 5)]
