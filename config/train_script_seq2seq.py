import ast
import logging

from config.ConfigHandler import ConfigHandler

BUCKETS = ast.literal_eval(ConfigHandler.get("bucket", "model_param"))

