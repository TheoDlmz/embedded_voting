# -*- coding: utf-8 -*-
"""
This file is part of Embedded Voting.
"""


def _cache(f):
    """
    Auxiliary decorator used by ``cached_property``.

    :param f: a method with no argument (except ``self``).
    :return: the same function, but with a `caching' behavior.
    """
    name = f.__name__

    # noinspection PyProtectedMember
    def _f(*args):
        try:
            return args[0]._cached_properties[name]
        except KeyError:
            # Not stored in cache
            value = f(*args)
            args[0]._cached_properties[name] = value
            return value
        except AttributeError:
            # cache does not even exist
            value = f(*args)
            args[0]._cached_properties = {name: value}
            return value

    _f.__doc__ = f.__doc__
    return _f


def cached_property(f):
    """
    Decorator used in replacement of @property to put the value in cache automatically.

    The first time the attribute is used, it is computed on-demand and put in cache. Later accesses to the
    attributes will use the cached value.

    Cf. :class:`DeleteCacheMixin` for an example.
    """
    return property(_cache(f))


class DeleteCacheMixin:
    """
    Mixin used to delete cached properties.

    Cf. decorator :meth:`cached_property`.

    >>> class Example(DeleteCacheMixin):
    ...     @cached_property
    ...     def x(self):
    ...         print('Big computation...')
    ...         return 6 * 7
    >>> a = Example()
    >>> a.x
    Big computation...
    42
    >>> a.x
    42
    >>> a.delete_cache()
    >>> a.x
    Big computation...
    42
    """

    def delete_cache(self) -> None:
        # noinspection PyAttributeOutsideInit
        self._cached_properties = dict()
