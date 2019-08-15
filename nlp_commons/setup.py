# Copyright (C) 2007-2011 Franco M. Luque
# URL: <http://www.cs.famaf.unc.edu.ar/~francolq/>
# For license information, see LICENSE.txt

from distutils.core import setup

setup(name='lq-nlp-commons',
         # Read the following page for advice on version numbering:
         # http://docs.python.org/distutils/setupscript.html#additional-meta-data
         version='0.2.0',
         description="Franco M. Luque's Common Python Code for NLP",
         author='Franco M. Luque',
         author_email='francolq@famaf.unc.edu.ar',
         url='http://www.cs.famaf.unc.edu.ar/~francolq/',
         packages=['dep'],
         py_modules=['bracketing', 'lbranch', 'paramdict',
            'treebank', 'wsj10', 'cast3lb', 'model', 'rbranch',
            'ubound', 'cast3lb10', 'negra', 'sentence', 'util',
            'eval', 'negra10', 'setup', 'wsj', 'graph'],
         license='GNU General Public License',
        )
