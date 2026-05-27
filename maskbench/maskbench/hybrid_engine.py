from .engine import Engine
from .llg_engine import LlgRawBytesEngine
from .xgr_engine import XgrEngine


UNSAFE_XGRAMMAR_KEYWORDS = {
    "$ref",
    "allOf",
    "anyOf",
    "contains",
    "dependencies",
    "dependentRequired",
    "dependentSchemas",
    "exclusiveMaximum",
    "exclusiveMinimum",
    "format",
    "if",
    "maxContains",
    "maxItems",
    "maxLength",
    "maximum",
    "minContains",
    "minItems",
    "minLength",
    "minimum",
    "multipleOf",
    "not",
    "oneOf",
    "pattern",
    "patternProperties",
    "propertyNames",
    "then",
    "uniqueItems",
}

ALLOWED_TYPES = {"array", "boolean", "integer", "null", "number", "object", "string"}


def _walk_schema(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_schema(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_schema(child)


def _is_object_schema(node: dict) -> bool:
    return node.get("type") == "object" or "properties" in node or "required" in node


def is_xgrammar_safe_subset(schema: dict) -> bool:
    for node in _walk_schema(schema):
        if any(keyword in node for keyword in UNSAFE_XGRAMMAR_KEYWORDS):
            return False

        schema_type = node.get("type")
        if schema_type is not None:
            if not isinstance(schema_type, str) or schema_type not in ALLOWED_TYPES:
                return False

        additional_properties = node.get("additionalProperties")
        if additional_properties is not None and additional_properties is not False:
            return False

        if _is_object_schema(node):
            properties = node.get("properties") or {}
            if not isinstance(properties, dict):
                return False
            required = node.get("required") or []
            if not isinstance(required, list):
                return False
            if not set(required).issubset(properties):
                return False
            if set(properties) - set(required):
                return False

    return True


class LlgXgrHybridEngine(Engine):
    def __init__(self):
        super().__init__()
        self.llg = LlgRawBytesEngine()
        self.xgr = XgrEngine()
        self.xgr.compliant = True
        self.active = None
        self.backend = ""

    def get_id(self):
        return "llg-xgr-hybrid"

    def get_name(self):
        return "LLGuidance/XGrammar hybrid"

    def get_module(self):
        return "maskbench"

    def get_version(self):
        return "local"

    def _configure_child(self, child: Engine):
        child.tokenizer = self.tokenizer
        child.tokenizer_model_id = self.tokenizer_model_id
        child.multi = self.multi
        child.debug = self.debug

    def init(self):
        self._configure_child(self.llg)
        self._configure_child(self.xgr)
        self.llg.init()
        self.xgr.init()

    def compile_grammar(self, schema: dict):
        if is_xgrammar_safe_subset(schema):
            self.active = self.xgr
            self.backend = "xgr-compliant"
        else:
            self.active = self.llg
            self.backend = "llg-rawbytes"
        self.active.compile_grammar(schema)

    def reset(self):
        self.active.reset()

    def compute_mask(self):
        self.active.compute_mask()

    def commit_token(self, token: int) -> bool:
        return self.active.commit_token(token)
