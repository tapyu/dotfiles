function nodeTranslation = translateXor(this, nodeTranslation, node, ~)
%   Copyright 2025 The MathWorks, Inc.

nodeTranslation = translateUniBroadcastPrefixBinaryOp(this, nodeTranslation, node, 'xor', []);
end