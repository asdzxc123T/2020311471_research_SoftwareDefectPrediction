## 데이터 스키마 규약(초안)

이 문서는 제가 커밋 마이닝, SZZ 라벨링, 모델 학습을 하나의 파이프라인으로 연결하기 위해 정의한 데이터 스키마를 정리한 것입니다.

**예측 단위(unit of prediction)**는 **함수(function) 수준**으로 두었습니다. SZZ는 보통 파일/라인 수준에서 결함 유발 변경을 식별하므로, 함수 단위 예측을 위해 **line-to-function 매핑**을 사용합니다. 구체적으로는 commit diff에서 변경된 라인을 추출한 뒤, 해당 라인이 속한 함수 범위를 정적 분석으로 식별하는 휴리스틱을 적용하고, 그 결과를 한 레코드(함수 단위)로 저장합니다.  
저장 단위는 기본적으로 `commit_file`을 확장한 **함수 단위**이며, 아래 필드로 식별합니다.

### 공통 키(식별자)

- `repo_url` (str): 원격 저장소 URL
- `repo_name` (str): 저장소 식별자(예: owner/name)
- `commit_hash` (str): 대상 커밋 SHA
- `commit_time` (datetime): 커밋 시간(Temporal split 기준)

### 변경 단위(함수 수준)

- `file_path` (str): 커밋 시점의 파일 경로
- `function_id` (str|None): 함수 식별자(파일 경로 + 함수 이름/시작 라인 등으로 구성, line-to-function 매핑 결과)
- `function_name` (str|None): 함수 이름(확장용)
- `changed_lines` (list|None): 해당 함수에서 변경된 라인 번호(매핑 시 사용)
- `language` (str|None): 추정 언어(확장용)

### 라벨(SZZ)

SZZ 계열 방법으로 BFC를 식별하고 git blame 역추적으로 BIC를 찾되, 제안서대로 리팩토링 탐지(renames, file moves), 코드 포맷 수정 필터, 이슈 번호–커밋 매칭 등 최소한의 휴리스틱을 적용해 오탐을 줄였습니다. 무작위 표본 수동 검증으로 라벨 정밀도(Precision)를 보고합니다.

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

