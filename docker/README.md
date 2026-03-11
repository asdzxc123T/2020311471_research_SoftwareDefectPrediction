## Docker 재현성 가이드

이 문서에서는 제 실험 환경을 Docker를 통해 재현하는 방법과, Joern/Defects4J와 같은 외부 도구를 어떤 방식으로 통합하려고 설계했는지 설명합니다.

### 기본(Trainer)

- `trainer` 서비스는 Python 기반 실험을 실행하기 위한 기본 컨테이너입니다.
- 리포지토리 루트에서 다음 명령으로 환경을 띄울 수 있습니다.

```bash
docker compose -f docker/docker-compose.yml up -d --build
docker compose -f docker/docker-compose.yml exec trainer pip install -e .
```

### Joern(선택)

제안서에서 요구하는 CPG 기반 구조 정보 추출을 위해, 저는 Joern을 별도 서비스로 두고 필요할 때만 활성화할 수 있도록 구성했습니다. 실제 연구에서는 다음 요소들을 명시적으로 고정하는 것이 중요합니다.

- Joern **버전 고정**(release tag)
- 대상 언어(예: Java) 및 입력 규모
- 산출물 포맷/저장 경로

현재 `docker/docker-compose.yml`에는 `joern` 서비스가 정의되어 있으며, 필요할 때만 켜도록 `profiles: ["optional"]`로 분리해 두었습니다.

```bash
docker compose -f docker/docker-compose.yml --profile optional up -d joern
docker compose -f docker/docker-compose.yml exec joern joern --version
```

> 참고: `joernio/joern:latest`는 환경에 따라 태그/이미지 제공이 변동될 수 있으므로, 실제 수행 시에는 특정 태그로 고정하는 것을 권장합니다.

### Defects4J(선택)

Defects4J는 의존성이 많기 때문에, 저는 Docker 환경에서 버전과 의존성을 고정하는 방향을 염두에 두고 스크립트 자리를 마련했습니다. 실제 활용 시에는 다음 항목을 구체적으로 결정해야 합니다.

- Java 버전, perl, build 도구(gradle/maven/ant)
- Defects4J 버전(tag) 및 설치 경로

