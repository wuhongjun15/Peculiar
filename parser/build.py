# Copyright (c) NUDT.

from tree_sitter import Language, Parser

Language.build_library(
  'my-languages.so',
  [
    'tree-sitter-solidity',
  ]
)

