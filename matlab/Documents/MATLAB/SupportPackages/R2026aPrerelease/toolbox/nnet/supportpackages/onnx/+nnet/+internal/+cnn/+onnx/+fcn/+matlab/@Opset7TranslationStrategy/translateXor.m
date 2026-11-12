function nodeTranslation = translateXor(this, nodeTranslation, node, ~)
%   Copyright 2025 The MathWorks, Inc.

nodeTranslation = translatePrefixBinaryOperator(this, nodeTranslation, node, 'xor');
end