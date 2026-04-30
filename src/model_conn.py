import os
from catboost import CatBoostClassifier
import pandas as pd
from sqlalchemy import create_engine
from typing import List
from fastapi import FastAPI
import hashlib

from datetime import datetime
from pydantic import BaseModel
from functools import lru_cache
from loguru import logger
from dotenv import load_dotenv



app = FastAPI()

def get_model_path(model_version: str) -> str:
    if (os.environ.get("IS_LMS") == "1"):
        model_path = f"/workdir/user_input/model_{model_version}"
    else:
        model_path = (
            f"/Users/lefukuro/Documents/recommendation_system/model/{model_version}"
        )
    return model_path

@lru_cache()
def load_models(model_version: str):
    model_path = get_model_path(model_version)
    model = CatBoostClassifier()
    model.load_model(model_path)

    return model

logger.info('loading model')
model_control = load_models("control")
model_test = load_models("test")

SALT = 'meow'
def get_user_group(id: int) -> str:
    value_str = str(id) + SALT
    value_num = int(hashlib.md5(value_str.encode()).hexdigest(), 16)
    percent = value_num % 100
    if percent < 50:
        return "control"
    elif percent < 100:
        return "test"
    return "unknown"

load_dotenv(dotenv_path='notebooks/.env', override=True)
@lru_cache()
def batch_load_sql(query: str) -> pd.DataFrame:
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("POSTGRES_HOST")
    port = os.getenv("POSTGRES_PORT")
    db = os.getenv("POSTGRES_DATABASE")

    url = f"postgresql://{user}:{password}@{host}:{port}/{db}"

    engine = create_engine(url)

    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=200000):
        chunks.append(chunk_dataframe)
        logger.info(f'got chunck: {len(chunk_dataframe)}')
    conn.close()
    return pd.concat(chunks, ignore_index=True)

@lru_cache()
def load_features_control() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info('loading liked posts')
    liked_posts = batch_load_sql("SELECT DISTINCT post_id, user_id FROM public.feed_data WHERE action='like'")

    logger.info('loading posts users features')
    posts_features = batch_load_sql("SELECT * FROM public.posts_info_features_dl")

    logger.info('loading users features')
    users_features = batch_load_sql("SELECT * FROM public.user_data")

    logger.info('loading posts text')
    posts_text = batch_load_sql("SELECT * FROM public.post_text_df")
    

    return liked_posts, users_features, posts_features, posts_text

@lru_cache()
def load_features_test() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info('loading liked posts')
    liked_posts = batch_load_sql("SELECT DISTINCT post_id, user_id FROM public.feed_data WHERE action='like'")

    logger.info('loading posts features')
    posts_features = batch_load_sql("SELECT * FROM public.posts_features_morozova_ekaterina")
    
    logger.info('loading users features')
    users_features = batch_load_sql("SELECT * FROM public.users_features_morozova_ekaterina")

    return liked_posts, posts_features, users_features

logger.info('service is up and running')

class PostGet(BaseModel):
    id: int
    text: str
    topic: str

    class Config:
        from_attributes = True

class Response(BaseModel):
    exp_group: str
    recommendations: List[PostGet]

def get_recommended_feed(
        id: int, 
        time: datetime, 
        limit: int = 5) -> Response:

    user_group = get_user_group(id)
    logger.info(f'user group={user_group}')
    
    if user_group == 'control':
        features = load_features_control()
        model = model_control

        logger.info(f'user id={id}')
        logger.info('reading features')
        user_features = features[1].loc[features[1].user_id == id]
        user_features = user_features.drop(['user_id'], axis=1)

        logger.info('dropping columns')
        posts_features = features[3].drop(['text', 'topic'], axis=1)
        posts_features = features[2].drop('index', axis=1)
        content = features[3][['post_id', 'text', 'topic']]

        logger.info('zipping everything')
        add_users_features = dict(zip(user_features.columns, user_features.values[0]))
        logger.info('assigning everything')
        user_posts_features = posts_features.assign(**add_users_features)
        user_posts_features = user_posts_features.set_index('post_id')

        logger.info('add time info')
        user_posts_features['hour'] = time.hour
        user_posts_features['month'] = time.month

        logger.info('predicting')
        user_posts_features = user_posts_features[model.feature_names_]
        predicts = model.predict_proba(user_posts_features)[:, 1]
        user_posts_features['predicts'] = predicts

    else:
        features = load_features_test() 
        model = model_test

        logger.info(f'user id={id}')
        logger.info('reading features')
        user_features = features[2].loc[features[2].user_id == id]
        user_features = user_features.drop('user_id', axis=1)

        logger.info('dropping columns')
        posts_features = features[1].drop(['text', 'topic'], axis=1)
        content = features[1][['post_id', 'text', 'topic']]

        logger.info('zipping everything')
        add_users_features = dict(zip(user_features.columns, user_features.values[0]))
        logger.info('assigning everything')
        user_posts_features = posts_features.assign(**add_users_features)
        user_posts_features = user_posts_features.set_index('post_id')

        logger.info('add time info')
        user_posts_features['hour'] = time.hour
        user_posts_features['month'] = time.month

        logger.info('predicting')
        predicts = model.predict_proba(user_posts_features)[:, 1]
        user_posts_features['predicts'] = predicts


    logger.info('deleting like posts')
    liked_posts = features[0]
    liked_posts = liked_posts[liked_posts.user_id == id].post_id.values
    _filtered = user_posts_features[~user_posts_features.index.isin(liked_posts)]

    recommended_posts = _filtered.sort_values('predicts')[-limit:].index
    
    recommendations = [
        PostGet(**{
            'id': int(i),
            'text': content[content.post_id == i].text.values[0],
            'topic': content[content.post_id == i].topic.values[0]
        }) 
        for i in recommended_posts
    ]
    return Response(exp_group=user_group, recommendations=recommendations)
    
@app.get("/post/recommendations/", response_model=Response)
def recommended_posts(
        id: int, 
        time: datetime, 
        limit: int = 5) -> Response:
    return get_recommended_feed(id, time, limit)