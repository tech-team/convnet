import json
import logging

import tornado
import tornado.web
from tornado.web import MissingArgumentError

from convnet_web import settings


class BaseHandler(tornado.web.RequestHandler):
    def __init__(self, application, request, **kwargs):
        super(BaseHandler, self).__init__(application, request, **kwargs)
        self.context = {}

    def prepare(self):
        self.context = {
            'debug': settings.DEBUG
        }

        if "Content-Type" in self.request.headers \
                and self.request.headers["Content-Type"].startswith("application/json"):
            try:
                self.request.body_json = json.loads(self.request.body, encoding='utf-8')
            except Exception as e:
                logging.warn("JSON parse error: %s", e.message)
                self.set_status(400)
                self.finish("JSON is malformed")

    def get_arg(self, field, default=None, field_type=None):
        args = self.request.body_json
        if args is None:
            args = self.request.body

        field_chain = field.split('.')
        target_args = args
        target_field = field_chain[0]
        if len(field_chain) != 1:
            for i, field in enumerate(field_chain):
                target_field = field
                if target_field in target_args:
                    if i != len(field_chain) - 1:
                        target_args = target_args[target_field]
                    continue
                else:
                    raise tornado.web.MissingArgumentError('.'.join(field_chain[:i + 1]))
        if target_field in target_args:
            value = target_args[target_field]
            if field_type is not None:
                return field_type(value)
            else:
                return value

        if field_type is not None:
            return field_type(default)
        return default

    def require_arg(self, field, field_type=None):
        arg = self.get_arg(field, field_type=field_type)
        if arg is None:
            raise tornado.web.MissingArgumentError(field)

    def write_json(self, data):
        self.set_header("Content-Type", "application/json")
        self.finish(json.dumps(data, ensure_ascii=False))

    def render(self, template_name, **kwargs):
        for k, v in self.context.iteritems():
            if k not in kwargs:
                kwargs[k] = v
        return super(BaseHandler, self).render(template_name, **kwargs)

