===============
Embedded Voting
===============


.. image:: https://img.shields.io/pypi/v/embedded_voting.svg
        :target: https://pypi.python.org/pypi/embedded_voting

.. image:: https://img.shields.io/travis/TheoDlmz/embedded_voting.svg
        :target: https://travis-ci.org/TheoDlmz/embedded_voting

.. image:: https://readthedocs.org/projects/embedded-voting/badge/?version=latest
        :target: https://embedded-voting.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


.. image:: https://codecov.io/gh/TheoDlmz/embedded_voting/branch/master/graphs/badge.svg
        :target: https://codecov.io/gh/TheoDlmz/embedded_voting/branch/master/graphs/badge
        :alt: Code Coverage





This contains the code for the work on embedded voting done during my internship at Nokia


* Free software: GNU General Public License v3
* Documentation: https://embedded-voting.readthedocs.io.


Features
--------

* Create a voting profile in which voters are associated to embeddings.
* Run elections on these profiles with different rules, using the geometrical aspects of the embeddings.
* The rules are defined for cardinal preferences, but some of them are adapted for the case of ordinal preferences.
* There are rules for single-winner elections and multi-winner elections.
* Classes to analyse the evolution of the score when the embeddings of one voter are changing.
* Classes to analyse the manipulability of the rules.
* Classes for algorithm aggregation.
* A lot of tutorials.

Credits
-------

This package was created with Cookiecutter_ and the `francois-durand/package_helper`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`francois-durand/package_helper`: https://github.com/francois-durand/package_helper
