## Temporal split 규약

이 문서에서는 제가 데이터 누수(data leakage)를 방지하기 위해 사용한 시간 기반 분할(Temporal split) 규약을 정리합니다.  
본 연구에서는 **시간 순서를 엄격히 준수하는 Temporal split**을 데이터 분할의 기본 원칙으로 삼았습니다.

### 기준 필드

- 기준 시간은 `commit_time`(UTC)로 고정한다.
- 동일한 `repo_name` 안에서 시간 순서를 유지하며 split을 적용한다.

### 기본 분할 규칙(권장)

- `train`: 가장 과거 구간
- `val`: 중간 구간
- `test`: 가장 미래 구간

구체적 경계는 설정 파일에서 `train_ratio`, `val_ratio`, `test_ratio`로 지정하거나, `cutoff_time`(날짜 경계)로 지정한다.

### 누수 방지 원칙

- `test` 기간의 정보를 이용해 `train/val`의 피처를 계산하지 않는다.
  - 예: 전역 IDF/정규화 파라미터/피처 스케일러는 `train`에서만 학습 후 `val/test`에 적용
- 동일 커밋에서 변경된 모든 단위(파일·함수)는 **반드시 같은 split**에 넣었습니다. 즉, 같은 커밋에 속한 함수들은 train/val/test가 서로 섞이지 않도록 관리합니다.

### 예외/경계 케이스 처리

- `commit_time`이 누락된 레코드는 제외하거나, 별도 split(`unknown_time`)로 격리한다.
- 동일 타임스탬프 다수 레코드는 (commit hash 정렬 등) 안정적 정렬 규칙을 적용한다.

