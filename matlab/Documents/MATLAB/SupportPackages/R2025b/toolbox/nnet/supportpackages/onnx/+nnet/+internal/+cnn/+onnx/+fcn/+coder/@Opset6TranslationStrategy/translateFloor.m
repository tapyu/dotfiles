function nodeTranslation = translateFloor(this, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2024 The MathWorks, Inc.

nodeTranslation = translateUnaryOperator(this, nodeTranslation, node, 'floor', IntegerTensorNames);
end

