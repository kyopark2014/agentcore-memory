# Agentcore Memory

AWS Bedrock AgentCore Memory를 활용하여 Agent에서 단기/장기 메모리를 활용하는 방법에 대해 설명합니다.

## 사전 설치 (Prerequisites)

```bash
python installer.py --project-name agentcore-memory --region us-west-2
```

`installer.py`가 수행하는 작업:
1. AgentCore Memory용 IAM Role 생성
2. AgentCore Memory 인스턴스 생성 (기존 memory가 있으면 건너뜀)
3. `application/config.json` 생성

## Memory 초기화 흐름

`chat.py`의 `initiate_memory()`가 최초 대화 시 memory를 초기화합니다.

```
save_to_memory() 호출
  └─ memory_id가 None이면 → initiate_memory()
       ├─ agentcore_memory.load_memory_variables(user_id)
       │    ├─ user_{user_id}.json 에서 기존 변수 로드
       │    ├─ memory_id가 없으면 → retrieve_memory_id()로 조회
       │    └─ 여전히 없으면 → create_memory()로 신규 생성
       ├─ create_strategy_if_not_exists()로 strategy 확인/생성
       └─ update_memory_variables()로 user_{user_id}.json에 저장
```

## 주요 변수

### config.json (프로젝트 설정)

| 변수 | 설명 |
|------|------|
| `projectName` | 프로젝트 이름. memory name으로도 사용 (`-` → `_` 치환) |
| `region` | AWS 리전 |
| `accountId` | AWS 계정 ID |
| `agentcore_memory_role` | Memory 실행용 IAM Role ARN |

### user_{user_id}.json (사용자별 설정)

| 변수 | 설명 |
|------|------|
| `memory_id` | AgentCore Memory 인스턴스 ID |
| `actor_id` | 사용자 식별자 (기본값: `user_id`) |
| `session_id` | 세션 ID (기본값: `uuid4().hex`로 자동 생성) |
| `namespace` | memory namespace (기본값: `/users/{actor_id}`) |

### chat.py 변수

| 변수 | 설명 |
|------|------|
| `agent_type` | 에이전트 타입 (기본값: `langgraph`) |
| `enable_memory` | memory 사용 여부 (`Enable` / `Disable`) |
| `user_id` | 사용자 ID (기본값: `agent_type` 값 사용) |

## 대화 저장 방법

`save_to_memory(query, result)`가 대화를 memory에 저장합니다.

```python
agentcore_memory.save_conversation_to_memory(memory_id, actor_id, session_id, query, result)
```

저장 과정:
1. `query`(사용자 입력)와 `result`(어시스턴트 응답)의 유효성 검사
2. AWS Bedrock 제한에 맞게 9,000자 초과 시 truncate
3. `(query, "USER")`, `(result, "ASSISTANT")` 형태의 conversation으로 변환
4. `memory_client.create_event()`로 memory에 이벤트 저장

### Memory Strategy

memory 생성 시 `customMemoryStrategy`가 함께 설정됩니다:
- **모델**: `us.anthropic.claude-haiku-4-5-20251001-v1:0`
- **역할**: 대화에서 사용자의 명시적/암시적 선호도를 추출
- **만료**: 이벤트 365일 후 만료


## MCP를 이용한 메모리 활용

MCP(Model Context Protocol) 서버를 통해 에이전트가 단기/장기 메모리에 접근할 수 있습니다. 각 MCP 서버는 `stdio` transport로 실행되며, `mcp_config.py`에서 설정을 관리합니다.

### 단기 메모리 (Short-Term Memory)

**파일**: `mcp_server_short_term_memory.py`

현재 세션의 최근 대화 이벤트를 조회하는 MCP 서버입니다.

**MCP 설정**:
```json
{
  "mcpServers": {
    "short-term memory": {
      "command": "python",
      "args": ["application/mcp_server_short_term_memory.py"]
    }
  }
}
```

**제공 도구**:

| 도구 | 파라미터 | 설명 |
|------|---------|------|
| `list_events` | `max_results` (기본값: 10) | 현재 세션의 최근 대화 이벤트 목록 조회 |

**동작 방식**:
1. `mcp.env`에서 `user_id`를 읽어옴
2. `agentcore_memory.load_memory_variables()`로 `memory_id`, `actor_id`, `session_id` 로드
3. `MemoryClient.list_events()`로 최근 대화 이벤트 반환

### 장기 메모리 (Long-Term Memory)

**파일**: `mcp_server_long_term_memory.py` → `mcp_long_term_memory.py`

대화에서 추출된 memory record를 저장, 검색, 조회, 삭제하는 MCP 서버입니다. 단기 메모리(이벤트)와 달리, 장기 메모리는 strategy에 의해 추출된 구조화된 기억입니다.

**MCP 설정**:
```json
{
  "mcpServers": {
    "long-term memory": {
      "command": "python",
      "args": ["application/mcp_server_long_term_memory.py"]
    }
  }
}
```

**제공 도구**:

| 도구 | 설명 |
|------|------|
| `long_term_memory` | 장기 메모리에 대한 CRUD 및 검색 |

**`long_term_memory` 파라미터**:

| 파라미터 | 필수 | 설명 |
|---------|------|------|
| `action` | O | 수행할 작업: `record`, `retrieve`, `list`, `get`, `delete` |
| `content` | record 시 | 저장할 텍스트 내용 |
| `query` | retrieve 시 | 시맨틱 검색 쿼리 |
| `memory_record_id` | get/delete 시 | 특정 memory record의 ID |
| `max_results` | - | 반환할 최대 결과 수 (기본값: 10) |
| `next_token` | - | 페이지네이션 토큰 |

**action별 동작**:

| action | 설명 | API |
|--------|------|-----|
| `record` | 텍스트를 memory에 이벤트로 저장 | `create_event()` |
| `retrieve` | 시맨틱 검색으로 관련 memory record 조회 | `retrieve_memory_records()` |
| `list` | 전체 memory record 목록 조회 | `list_memory_records()` |
| `get` | 특정 memory record를 ID로 조회 | `get_memory_record()` |
| `delete` | 특정 memory record 삭제 | `delete_memory_record()` |

### 단기 vs 장기 메모리 비교

| 구분 | 단기 메모리 | 장기 메모리 |
|------|-----------|-----------|
| 데이터 | 원본 대화 이벤트 (USER/ASSISTANT) | strategy가 추출한 구조화된 기억 |
| 범위 | 현재 세션의 최근 대화 | 세션/시간에 걸친 축적된 지식 |
| 검색 | 시간순 목록 조회 | 시맨틱 검색 지원 |
| 용도 | 최근 맥락 참조 | 사용자 선호도, 패턴 등 장기 기억 활용 |

### 공통 사전 조건

두 MCP 서버 모두 `mcp.env` 파일에서 `user_id`를 읽어 사용자를 식별합니다:

```json
{"user_id": "langgraph"}
```
