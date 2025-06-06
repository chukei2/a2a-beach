# Airbnb Agent - LangGraphベースのAIエージェント

## 概要

AirbnbAgentは、Airbnb宿泊施設の検索と関連する質問に特化したAIエージェントです。LangGraphとGoogle GenerativeAI（Gemini）を使用して構築されており、MCPツールを活用してAirbnb物件の検索機能を提供します。

## 主な機能

- **Airbnb物件検索**: MCPツールを使用した宿泊施設の検索
- **対話型応答**: ユーザーとの自然な会話形式での情報提供
- **ストリーミング対応**: リアルタイムでの応答生成
- **セッション管理**: メモリ機能による会話履歴の保持
- **エラーハンドリング**: 適切なエラー処理と回復機能
- **フェイルオーバー**: MCPツールが読み込めない場合はローカル検索ツールに自動切り替え

## アーキテクチャ

### コアコンポーネント

#### 1. `AirbnbAgent`クラス
- **役割**: メインのエージェントクラス
- **機能**: 
  - Google Gemini 2.5 Flash PreviewモデルとのインターフェースMCP
  - LangGraphのReactエージェントパターンの実装
  - セッション管理とメモリ保存

#### 2. `ResponseFormat`クラス
```python
class ResponseFormat(BaseModel):
    status: Literal["input_required", "completed", "error"]
    message: str
```
- **役割**: 構造化された応答フォーマット
- **ステータス**:
  - `input_required`: ユーザーからの追加入力が必要
  - `completed`: リクエスト完了
  - `error`: エラー発生

### システム設計

#### エージェントの動作フロー
1. **初期化**: MCPツールとGeminiモデルの設定
2. **クエリ受信**: ユーザーからの検索リクエスト
3. **ツール実行**: MCPツールを使用したAirbnb API呼び出し
4. **応答生成**: 構造化された形式での結果返却

#### メモリ管理
- `MemorySaver`を使用したセッション状態の永続化
- スレッドIDベースの会話履歴管理

## API仕様

### 主要メソッド

#### `ainvoke(query: str, sessionId: str) -> dict[str, Any]`
**非同期でクエリを実行し、完全な応答を返す**

**パラメータ:**
- `query`: 検索クエリ（例：「東京のAirbnb物件を探して」）
- `sessionId`: セッション識別子

**戻り値:**
```python
{
    "is_task_complete": bool,    # タスク完了状態
    "require_user_input": bool,  # 追加入力の必要性
    "content": str               # 応答メッセージ
}
```

#### `stream(query: str, sessionId: str) -> AsyncIterable[Any]`
**ストリーミング形式でリアルタイム応答を生成**

**パラメータ:**
- `query`: 検索クエリ
- `sessionId`: セッション識別子

**戻り値:** 
- 非同期イテレータによる段階的な応答チャンク

## セットアップと使用方法

### 前提条件
- Python 3.9以上
- Google Cloud APIキー
- 必要なPythonパッケージ（`pyproject.toml`参照）

### インストール手順

1. **環境設定**
   ```bash
   # APIキーの設定
   echo "GOOGLE_API_KEY=your_api_key_here" > .env
   ```

2. **依存関係のインストール**
   ```bash
   # uvを使用した場合
   uv sync
   
   # pipを使用した場合
   pip install -r requirements.txt
   ```

3. **サーバー起動**
   ```bash
   uv run .
   ```

### 使用例

#### 基本的な使用方法
```python
from airbnb_agent.agent import AirbnbAgent

# MCPツールの設定（実際の実装に応じて）
mcp_tools = [...]  # Airbnb検索ツール

# エージェント初期化
agent = AirbnbAgent(mcp_tools=mcp_tools)

# 同期呼び出し
response = await agent.ainvoke(
    query="東京の2名用の部屋を検索して",
    sessionId="user_session_123"
)

print(response["content"])
```

#### ストリーミング使用例
```python
async for chunk in agent.stream(
    query="渋谷近くのAirbnb物件を教えて",
    sessionId="user_session_123"
):
    print(chunk["content"])
```

## 設定

### システム指示（SYSTEM_INSTRUCTION）
エージェントは以下の指示で動作します：
- Airbnb宿泊施設専門のアシスタント
- 提供されたツールのみを使用
- 物件情報や価格の捏造禁止
- Markdown形式での応答
- 物件への直接リンク提供

### サポートするコンテンツタイプ
- `text`
- `text/plain`

## エラーハンドリング

### エラー分類
1. **HTTPエラー**: 外部API呼び出し失敗
2. **認証エラー**: APIキー関連の問題
3. **セッションエラー**: 無効なセッションID
4. **ツールエラー**: MCPツールの実行失敗

### エラー応答例
```python
{
    "is_task_complete": True,
    "require_user_input": False,
    "content": "申し訳ございませんが、現在Airbnb検索ツールが利用できません。後ほど再試行してください。"
}
```

## ログ機能

詳細なログ出力により、以下の情報を追跡可能：
- エージェント初期化状況
- クエリ実行プロセス
- ツール使用状況
- エラー発生時の詳細情報

### ログレベル
- `INFO`: 一般的な実行情報
- `DEBUG`: 詳細なデバッグ情報
- `WARNING`: 警告メッセージ
- `ERROR`: エラー情報

## 開発・カスタマイズ

### 拡張ポイント
1. **新しいMCPツールの追加**
2. **応答フォーマットのカスタマイズ**
3. **エラーハンドリングの強化**
4. **新しい検索フィルターの実装**

### テスト
```bash
# テストクライアントの実行
python test_client.py
```

## トラブルシューティング

### よくある問題

1. **APIキーエラー**
   - `.env`ファイルにGoogle APIキーが正しく設定されているか確認

2. **MCPツールエラー**
   - MCPツールが正しく初期化されているか確認
   - 必要な権限が設定されているか確認
   - 読み込みに失敗した場合は自動的にローカル検索ツールへフォールバック

3. **メモリ関連エラー**
   - セッションIDが有効な文字列として提供されているか確認

## ライセンス

このプロジェクトのライセンス情報については、プロジェクトルートの`LICENSE`ファイルを参照してください。
