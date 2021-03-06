import factory
from factory.fuzzy import FuzzyChoice

from grandchallenge.components.models import (
    ComponentInterface,
    ComponentInterfaceValue,
)


class ComponentInterfaceFactory(factory.DjangoModelFactory):
    class Meta:
        model = ComponentInterface

    title = factory.Sequence(lambda n: f"Component Interface {n}")
    kind = FuzzyChoice(ComponentInterface.Kind.values)
    default_value = None
    relative_path = factory.Sequence(lambda n: f"interface-{n}")


class ComponentInterfaceValueFactory(factory.DjangoModelFactory):
    class Meta:
        model = ComponentInterfaceValue

    interface = factory.SubFactory(ComponentInterfaceFactory)
