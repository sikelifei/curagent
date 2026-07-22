做完下面的实验
用本地的/data2/zhangwenjian/model/Qwen3-4B和instruct模型分别做实验，可以用vllm，myvllm的conda启动，openai的格式替换api的地址等，
第一个是/data2/zhangwenjian/agent/curagent  的webshop，跑200条测试，记录这个成功率和score reward，还有递归的触发次数，深度等值
第二个是oolong synth的实验，从8K到512K，每个做五十个，如果有就做，没有就做bucket的上限，然后1m到4m每个做二十个，记录处理耗时，还有成功率，递归发生情况
