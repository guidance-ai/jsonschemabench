
import uuid
import ipaddress
from typing import Dict, Any
import jsonschema
from jsonschema import Draft202012Validator, FormatChecker, ValidationError


def is_json_schema_valid(schema: dict):
    """
    Check if a JSON schema is valid.

    :param schema: A JSON schema.
    :return: True if the schema is valid, False otherwise.
    """
    try:
        # Check if the schema is valid
        jsonschema.Draft202012Validator.check_schema(schema)
        return True
    except jsonschema.SchemaError as e:
        # logger.error(f"SchemaError: The JSON schema is invalid. {e.message}")
        return False


# Initialize the FormatChecker
format_checker = FormatChecker()

# Add custom format checkers
@format_checker.checks("ipv4")
def ipv4_check(value):
    ipaddress.IPv4Address(value)


@format_checker.checks("ipv6")
def ipv6_check(value):
    ipaddress.IPv6Address(value)


@format_checker.checks("uuid")
def uuid_check(value):
    uuid.UUID(value)



def validate_enhanaced(instance: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """
    Validate a JSON instance against a schema with enhanced format checking.

    :param schema: The JSON schema to validate against.
    :param instance: The JSON instance to validate.
    :raises ValidationError: If the validation fails.
    """
    # first check if the schema is valid
    if not is_json_schema_valid(schema):
        raise ValidationError("The JSON schema is invalid.")
    validator = Draft202012Validator(schema, format_checker=format_checker)
    try :
        validator.validate(instance)
    except ValidationError as e:
        raise ValidationError(e.message)
    return True