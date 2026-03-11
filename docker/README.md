## Docker 재현성 가이드

### 기본(Trainer)

- `trainer`는 Python 실행/실험을 위한 기본 컨테이너입니다.
- 루트에서 실행:

```bash
docker compose -f docker/docker-compose.yml up -d --build
docker compose -f docker/docker-compose.yml exec trainer pip install -e .
```

### Joern(선택)

이 리포는 제안서 요구에 맞춰 Joern(CPG 생성) 단계를 포함하지만, 실제 연구에서는 다음을 고정하는 것이 중요합니다.

- Joern **버전 고정**(release tag)
- 대상 언어(예: Java) 및 입력 규모
- 산출물 포맷/저장 경로

현재 `docker/docker-compose.yml`에는 `joern` 서비스 자리가 있으며, 필요할 때만 켜도록 `profiles: ["optional"]`로 분리했습니다.

```bash
docker compose -f docker/docker-compose.yml --profile optional up -d joern
docker compose -f docker/docker-compose.yml exec joern joern --version
```

> 참고: `joernio/joern:latest`는 환경에 따라 태그/이미지 제공이 변동될 수 있으므로, 실제 수행 시에는 특정 태그로 고정하는 것을 권장합니다.

### Defects4J(선택)

Defects4J는 의존성이 많아 Docker에서 고정하는 것이 유리합니다. 본 뼈대에서는 우선 Trainer 컨테이너에 스크립트 자리를 마련하고, 실제 사용 시 다음을 확정합니다.

- Java 버전, perl, build 도구(gradle/maven/ant)
- Defects4J 버전(tag) 및 설치 경로

