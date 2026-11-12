function nodeTranslation = translateReduceMax(this, nodeTranslation, node, IntegerTensorNames)
    %   Copyright 2024 The MathWorks, Inc. 

    op = '%s = max(%s, [], dims);\n';
    nodeTranslation = translateReduceOp(this, nodeTranslation, node, IntegerTensorNames, op);
end
