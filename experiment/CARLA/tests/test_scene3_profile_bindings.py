from benchmark.catalog import load_catalog
from benchmark.task_oracle import load_profile
from benchmark.planning import build_plan


def test_new_scene3_profiles_bind_and_do_not_claim_full_coverage():
    catalog = load_catalog("scene_3")
    for task_id, kinds in (
        ("scene3_right_lane_closure", ["speed_ceiling", "lane_change"]),
        ("scene3_resume_normal_driving", ["speed_change", "destination"]),
    ):
        task = catalog.select(task_id)[0]
        profile = load_profile(catalog, task)
        assert profile is not None
        assert [step["kind"] for step in profile["steps"]] == kinds
        assert profile["coverage"]["status"] == "PARTIAL"
        assert profile["coverage"]["missing"]
        assert build_plan(catalog, task_id)["missing_oracle_profiles"] == []


def test_all_formal_tasks_have_source_bound_profiles():
    for scene, count in (("scene_1",15),("scene_2",15),("scene_3",8)):
        catalog=load_catalog(scene)
        assert len(catalog.tasks)==count
        assert build_plan(catalog,'all')['missing_oracle_profiles']==[]
