# MandelbrotV1_Simple — 分布式增量预训练项目轻量版

本项目是基于 MandelbrotV1 模型的分布式增量预训练框架，支持 Server / Middle Server / Client 三种角色，使用 gRPC 通信。

---

## 目录结构

MandelbrotV1_Simple/
├── .vscode/ # VS Code 配置（launch.json, settings.json）
├── .venv/ # Python 虚拟环境
├── scripts/ # 训练脚本
│ ├── train_pretrain_france_server.py # Server 端训练入口
│ ├── train_pretrain_france_middle_server.py # Middle Server 训练入口
│ └── train_pretrain_france_client.py # Client 端训练入口
├── mandelbrot_service.proto # gRPC 服务定义
├── mandelbrot_service_pb2.py # protobuf 生成
├── mandelbrot_service_pb2_grpc.py # gRPC 生成
├── requirements.txt # Python 依赖
├── init_vs_env.bat # VS 环境初始化脚本
├── tokenizers-0.19.1-cp312-cp312-win_amd64.whl # tokenizers 离线安装包
└── README.md

> 模型权重、Tokenizer、训练数据等位于上层的 `../MandelbrotV1_Simple/` 目录中。

---

## 环境配置

### 1. 创建虚拟环境并激活

```powershell
# 在项目根目录
python -m venv .venv
activate

2. 安装依赖
pip install -r requirements.txt

3. 安装分词器tokenizers==0.19.1
如果通过 pip install -r requirements.txt 在线安装失败（网络限制或平台不兼容），可以使用项目根目录下预置的离线 .whl 文件手动安装到.venv环境中：
D:/MandelbrotV1_Simple/.venv/Scripts/python.exe -m pip install tokenizers-0.19.1-cp312-cp312-win_amd64.whl

4. 修改.vscode文件下面的launch.json文件。
"workspaceFolder" 修改成当前项目的绝对路径，比如说我这个项目的路径是："D:/MandelbrotV1_Simple"。
如果是单机单卡跑程序，还要设置以下内容，即修改"CUDA_VISIBLE_DEVICES":，"RANK":，"WORLD_SIZE":, "LOCAL_RANK":这4个量，分别对应Server、Middle Server、Client的GPU ID，
"program": "${workspaceFolder}/scripts/train_pretrain_france_middle_server1.py", 要改成你想要运行的脚本，比如说"${workspaceFolder}/scripts/train_pretrain_france_middle_server.py"
        {
            "name": "预训练任务 GPU0_server",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/scripts/train_pretrain_france_server.py",
            "console": "integratedTerminal",
            "env": {
                 "CUDA_VISIBLE_DEVICES": "0", //"0,1,2,3,4,5",
                 "RANK": "0",
                 "WORLD_SIZE": "1", //"5",      
                 "LOCAL_RANK": "0" 
            },
            "justMyCode": true
        },
        {
            "name": "预训练任务 GPU1_middle",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/scripts/train_pretrain_france_middle_server.py",
            "console": "integratedTerminal",
            "preLaunchTask": "Delay 1 Seconds",  
            "env": {
                "CUDA_VISIBLE_DEVICES": "0", //"0,1,2,3,4,5",
                "RANK": "0",
                "WORLD_SIZE": "1", //"5",      
                "LOCAL_RANK": "0" 
            },
            "justMyCode": true
        },
        {
            "name": "预训练任务 GPU7_client",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/scripts/train_pretrain_france_client.py",
            "console": "integratedTerminal",
            "preLaunchTask": "Delay 4 Seconds",  
            "env": {
                 "CUDA_VISIBLE_DEVICES": "0", //"0,1,2,3,4,5",
                 "RANK": "0", //"4",
                 "WORLD_SIZE": "1", //"5",      
                 "LOCAL_RANK": "0" //"4"
            },
            "justMyCode": true
        },  

5. 把3个脚本train_pretrain_france_server，train_pretrain_france_middle_server，train_pretrain_france_client里的方法def parse_args():里面的路径改成你本地的路径，比如说，把
p.add_argument("--train_folder", type=str,default="../MandelbrotV1/data_france/france_en.txt", help="Folder containing .txt files (recursively)")
"../MandelbrotV1_Simple/france_en.txt" 改成你本地的路径，比如说："D:/MandelbrotV1_Simple/france_en.txt"

p.add_argument("--tokenizer_name", type=str, default="../MandelbrotV1_/trained_tokenizer_france", help="Tokenizer to use (default: gpt2)")的
"../MandelbrotV1_/trained_tokenizer_france"改成你本地的路径，比如说："D:/MandelbrotV1_Simple/trained_tokenizer_france"

即把所有关于路劲的超参数修改成本地的路径，例如以下这几个参数：
p.add_argument("--output_dir", type=str, ...)
p.add_argument("--model_name_or_path", type=str, ...)
p.add_argument("--init_from_dir", type=str, ...)
p.add_argument("--block_cache_dir", type=str, ...)
p.add_argument("--log_dir", type=str, ...)

## 运行说明
修改好这些配置之后，就可以在vscode的左边菜单里选择运行与调试，直接运行launch.json里的3个配置，先运行Server，再运行Middle Server，最后运行Client，就可以进行分布式增量预训练了。

