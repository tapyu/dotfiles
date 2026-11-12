function nodeTranslation = translateReduceMin(this, nodeTranslation, node, IntegerTensorNames)
    %   Copyright 2024 The MathWorks, Inc. 

    op = '%s = min(%s, [], dims);\n';
    nodeTranslation = translateReduceOp(this, nodeTranslation, node, IntegerTensorNames, op);
end