class AttributeHashmap(dict):
    """
    A Specialized hashmap such that:
        hash_map = AttributeHashmap(dictionary)
        `hash_map.key` is equivalent to `dictionary[key]`
    """
    def __init__(self, *args, **kwargs):
        super(AttributeHashmap, self).__init__(*args, **kwargs)
        self.__dict__ = self
