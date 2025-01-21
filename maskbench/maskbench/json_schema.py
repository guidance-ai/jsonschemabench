from .json_schema_to_grammar import SchemaConverter


def json_schema_to_gbnf(schema: dict):
    url = "stdin"
    converter = SchemaConverter(
        prop_order={}, allow_fetch=False, dotall=False, raw_pattern=False
    )
    xschema = converter.resolve_refs(schema, url)
    converter.visit(xschema, "")
    grm = converter.format_grammar()
    return grm
