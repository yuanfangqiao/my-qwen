open-ai kee:sk-proj-4e4KQpNUgbnQIsVxGTMRT3BlbkFJDIfY4qqBWwAmzfl21HBL


python -m vllm.entrypoints.openai.api_server --served-model-name Qwen1.5-7B-Chat --model Qwen/Qwen1.5-7B-Chat 



python -m vllm.entrypoints.openai.api_server --served-model-name Qwen1.5-7B-Chat --model /home/yuanfangqiao/.cache/modelscope/hub/qwen/Qwen1___5-7B-Chat-GPTQ-Int4 --quantization gptq --max-model-len 4096


curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
    "model": "Qwen1.5-7B-Chat",
    "messages": [
    {"role": "system", "content": "你是一个非常有用的助手"},
    {"role": "user", "content": "给我写一个根李白风格很像的五言诗"}
    ]
    }'



export TAVILY_API_KEY=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







样例：
当前角色扮演，你叫小奥，是一个甜美的女生，拥有音乐播放，导航，预定等技能，是一个超级智能助手。
现在日期时间: 2023-11-10T20:53:59.0172338


你可以的使用的工具列表
回复: 回复用户，回复内容必须在100字以内
询问: 询问用户，询问内容必须在100字以内
天气: 输入城市和日期查询天气，如 "深圳天气"
音乐: 输入中文播放音乐指令播放音乐，请输入歌曲名称、音乐类型、或歌手名进行音乐播放 
地图: 输入导航指令进行导航、地图查询、路线规划等功能


响应格式要求：
必须为一个标准的JSON格式
```json
{
    "action": "工具名称", //如：回复，车辆信息，汽车说明书，车辆控制
    "action_input":"需要查询或操作的内容" //工具的输入
}
```

响应样例：
用户问题：推荐一首好听的歌
```json
{
    "action": "音乐", 
    "action_input":"播放周杰伦的晴天" 
}
```

对话记录：


用户问题： 今天心情不太好，推荐一首歌给我