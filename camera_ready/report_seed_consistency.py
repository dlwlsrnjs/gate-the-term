"""Describe seed-1/2 consistency using validated, complete exported runs only.

Run report_completed.py first. Active runs contribute progress, never accuracy.
No partial-seed mean or training-seed confidence interval is estimated here.
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LANGS = ["bn", "te", "th"]
ARMS = ["vanilla", "split", "cgh_strict", "pivot_rsft", "cgh_legacy", "cgh_strict_nois"]
PAIRS = [("vanilla", "split"), ("vanilla", "cgh_strict"),
         ("split", "cgh_strict"), ("vanilla", "pivot_rsft"),
         ("split", "pivot_rsft"), ("pivot_rsft", "cgh_strict")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=HERE / "results/latest")
    args = parser.parse_args()
    report = json.loads((args.directory / "completed_results.json").read_text())
    progress = json.loads((args.directory / "progress.json").read_text())
    assert report["as_of_kst"] == progress["as_of_kst"]
    runs = {(r["arm"], r["seed"]): r for r in report["completed_runs"]}
    active = {w["job"]: w for w in progress["active"]}
    forecast = {f["job"]: f for f in progress["forecast"]}
    comparisons = {(r["baseline"], r["arm"], r["seed"]): r for r in report["per_seed_comparisons"]}
    lines = ["# Seed 1·2 일관성 확인 — 최종 평가 완료 기준", "",
             f"기준 시각: {report['as_of_kst']}", "",
             "완료된 100-step 최종 result.json을 검증한 집계만 성능으로 표시합니다. 학습 100 step과 언어별 평가 파일 일부만으로 전체 평가 완료를 선언하지 않습니다. 미완료/미취합 칸은 0점이 아닙니다.", "",
             "| 조건 | Seed 1: bn / te / th, 각 /60 | Seed 1: 전체 /180 | Seed 2: bn / te / th, 각 /60 | Seed 2: 전체 /180 | Seed 2 상태 |",
             "|---|---|---:|---|---:|---|"]
    for arm in ARMS:
        cells = []
        for seed in [1, 2]:
            r = runs.get((arm, seed))
            cells += [" / ".join(str(r["by_language"][lg]["corr"]) for lg in LANGS),
                      f"{r['overall']['corr']}/180 ({r['overall']['corr'] / 180:.2%})"] if r else ["미완료/미취합", "보류"]
        tag = f"{arm}_s2_n100"
        worker, estimate = active.get(tag), forecast.get(tag)
        state = "미완료/미취합"
        if (arm, 2) in runs:
            state = "학습·최종 평가 완료"
        elif worker:
            if worker["step"] == 100:
                languages = ", ".join(worker.get("evaluated_languages", [])) or "없음"
                state = f"학습 완료; 최종 평가 중 (완료 언어: {languages})"
            else:
                state = f"학습 {worker['step']}/100"
            if estimate:
                state += f"; 평가 포함 {estimate['finish_kst']} KST 예상"
        lines.append(f"| {arm} | " + " | ".join(cells) + f" | {state} |")
    lines += ["", "## 같은 seed의 대조 효과가 반복되는가", "",
              "| 비교 | Seed 1 차이 | Seed 2 차이 | 전체 차이의 방향만 비교 |",
              "|---|---:|---:|---|"]
    language_rows = []
    for left, right in PAIRS:
        values = []
        for seed in [1, 2]:
            c = comparisons.get((left, right, seed))
            values.append(c["delta_pp"] if c else None)
            if c:
                delta = [runs[right, seed]["by_language"][lg]["corr"] - runs[left, seed]["by_language"][lg]["corr"] for lg in LANGS]
                low, high = c["conditional_semantic_cluster_95ci_pp"]
                language_rows.append(f"| {right} − {left} | {seed} | " + " | ".join(f"{v:+d}" for v in delta)
                                     + f" | {c['delta_pp']:+.2f} | [{low:+.2f}, {high:+.2f}] |")
        if any(v is None for v in values):
            judgment = "두 seed의 직접 대조 미완료 — 보류"
        elif values == [0., 0.]:
            judgment = "두 seed 모두 총점 차이 0; 동등성 입증은 아님"
        elif all(v > 0 for v in values):
            judgment = "두 seed에서 모두 증가; 크기·언어별 일관성은 별도 확인"
        elif all(v < 0 for v in values):
            judgment = "두 seed에서 모두 감소; 크기·언어별 일관성은 별도 확인"
        else:
            judgment = "방향 불일치 또는 한 seed만 차이 0"
        cells = [f"{v:+.2f} pp" if v is not None else "보류" for v in values]
        lines.append(f"| {right} − {left} | " + " | ".join(cells) + f" | {judgment} |")
    lines += ["", "## 언어별 차이와 문항 불확실성", "",
              "언어별 칸은 같은 seed의 두 조건 간 정답 수 차이(/60), 전체는 percentage points입니다.", "",
              "| 비교 | Seed | Bengali Δ | Telugu Δ | Thai Δ | 전체 Δ pp | 조건부 95% 구간 pp |",
              "|---|---:|---:|---:|---:|---:|---|"] + language_rows
    lines += ["", "원문 ID 60개를 묶어 세 번역을 함께 resample한 탐색적 구간입니다(20,000회, RNG 3213). 학습 seed 변동과 다중 비교 보정을 포함하지 않습니다. 전체 입력 180개를 독립 문제로 세지 않습니다.", "",
              "같은 방법의 seed 1→2 절대 점수 변화는 seed 변동입니다. 방법의 효과는 각 seed에서 대응 기준선과 비교합니다. 총점이나 언어별 정답 수가 같아도 동일 문항을 맞혔다는 뜻은 아닙니다.", "",
              "## 해석 범위와 진행", "",
              "- Strict−pivot의 전체 차이뿐 아니라 bn/te/th 차이의 방향을 함께 확인합니다. 한 언어의 이득이 다른 언어의 손실을 상쇄할 수 있습니다.",
              "- 영어 접근 대조의 성능 차이를 순수 영어 내용 효과와 estimator 효과의 인과적 합으로 분해하지 않습니다.",
              "- 사전 계획한 세 seed가 모두 완료되지 않은 방법의 3-seed 평균은 보류합니다. 두 seed의 방향 일치만으로 재현성이나 우위를 확정하지 않습니다.", "",
              f"로컬 완료 {len(progress['completed_local'])}/18, 실행 {len(progress['active'])}, 대기 {len(progress['pending_local'])}, 실패 {len(progress['failed'])}. 다른 환경 배정은 후순위 학습 9회와 확장 평가 19개 모델이며 실제 원격 실행 상태는 이 서버에서 관측하지 않았습니다.", "",
              "[전체 완료 결과](RESULTS_KO.md) · [진행 및 완료 시각 추정](PROGRESS_KO.md) · [Seed 1 영어 정보 효과 분석](../CGH_PIVOT_KO.md) · [문항별 CSV](completed_items.csv) · [분석·출처 JSON](completed_results.json)", "",
              "재생성: 먼저 report_completed.py로 같은 시각의 결과·진행 스냅샷을 만들고, python camera_ready/report_seed_consistency.py를 실행합니다.", ""]
    (args.directory / "SEED_CONSISTENCY_KO.md").write_text("\n".join(lines))
    print(json.dumps({"as_of_kst": report["as_of_kst"], "requested_seed2_complete":
                      {arm: (arm, 2) in runs for arm in ["cgh_strict", "pivot_rsft"]}}))


if __name__ == "__main__":
    main()
