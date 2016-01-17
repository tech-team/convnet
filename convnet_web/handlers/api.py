from convnet_web.handlers import BaseHandler


class ApiHandler(BaseHandler):

    def api_respond(self, status='ok', data=None):
        d = dict(status=status)
        if data is not None:
            d['data'] = None
        return self.write_json(d)

    def api_ok(self, data):
        return self.api_respond('ok', data)

    def api_error(self, data):
        return self.api_respond('error', data)

    def require_arg(self, field, default=None, field_type=None):
        args = self.request.body_json
        if args is None:
            args = self.request.body
        if field in args:
            value = args[field]
            if field_type is not None:
                return field_type(value)
            else:
                return value

        if field_type is not None:
            return field_type(default)
        return default


class ApiConfig(ApiHandler):
    def post(self):
        req = self.request.body_json

        config = dict(
            learning_rate=self.require_arg('learning_rate', field_type=float),
            momentim=self.require_arg('momentum', field_type=float),
            batch_size=self.require_arg('batch_size', field_type=int),
            iterations_count=self.require_arg('iterations_count', field_type=int),
        )
        pass
