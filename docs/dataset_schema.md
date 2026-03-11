## 데이터 스키마 규약(초안)

이 문서는 제가 커밋 마이닝, SZZ 라벨링, 모델 학습을 하나의 파이프라인으로 연결하기 위해 정의한 데이터 스키마를 정리한 것입니다.  
데이터셋은 기본적으로 “변경 단위(change unit)”를 레코드로 사용하며, 기본 단위는 `commit_file`이고, 필요 시 `commit_hunk` 또는 `commit_function`으로 확장할 수 있도록 설계했습니다.

### 공통 키(식별자)

- `repo_url` (str): 원격 저장소 URL
- `repo_name` (str): 저장소 식별자(예: owner/name)
- `commit_hash` (str): 대상 커밋 SHA
- `commit_time` (datetime): 커밋 시간(Temporal split 기준)

### 변경 단위(기본: commit_file)

- `file_path` (str): 커밋 시점의 파일 경로
- `language` (str|None): 추정 언어(확장용)

### 라벨(SZZ)

- `is_bug_fixing_commit` (bool): BFC 여부(커밋 메시지/이슈 링크 기반)
- `is_bug_inducing` (bool): BIC로 추정되는 변경 단위 여부(SZZ 결과)
- `label_source` (str): 라벨 생성 방식(예: `szz_v1`, `manual_override`)
- `label_confidence` (float|None): 옵션(휴리스틱/검증 기반)

### 노력/정적 메트릭(effort-aware)

- `loc_added` (int)
- `loc_deleted` (int)
- `churn` (int): `loc_added + loc_deleted`
- `loc_current` (int|None): 변경 직후 파일 LOC(옵션)
- `cyclomatic_complexity` (float|None): 옵션(추후 확장)

### 텍스트/구조 입력

- `diff_text` (str|None): 변경 diff(모델 입력)
- `code_text` (str|None): 원문 코드/컨텍스트(옵션)
- `cpg_path` (str|None): Joern 산출물/그래프 경로(옵션)

### 분할(split)

- `split` (str): `train` / `val` / `test` (Temporal split 결과)

### 파일 포맷

- 1차 권장: **Parquet**
- 교환/검증용: CSV/JSONL

