import os
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
from bson import ObjectId

# 加载环境变量
load_dotenv()

# 连接MongoDB
uri = os.getenv("MONGODB_URI")
db_name = os.getenv("MONGODB_DB")
collection_name = os.getenv("MONGODB_COLLECTION")

client = MongoClient(uri, serverSelectionTimeoutMS=5000)
db = client[db_name]
collection = db[collection_name]

print(f"✅ Connected to database '{db_name}', collection '{collection_name}'")

# 查询所有文档
docs = list(collection.find({}))
print(f"📄 Fetched {len(docs)} documents")

# 将 ObjectId 转为字符串
def normalize(doc):
    return {k: str(v) if isinstance(v, ObjectId) else v for k, v in doc.items()}

data = [normalize(d) for d in docs]
df = pd.DataFrame(data)

# 保存到 CSV 文件
output_path = f"{collection_name}.csv"
df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"✅ Exported data to {output_path}")

# 打印前 5 行数据作为预览
print(df.head())
