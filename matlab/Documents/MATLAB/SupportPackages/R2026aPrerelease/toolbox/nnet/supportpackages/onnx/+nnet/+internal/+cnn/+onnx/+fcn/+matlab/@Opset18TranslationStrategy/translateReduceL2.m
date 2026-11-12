function nodeTranslation = translateReduceL2(this, nodeTranslation, node, IntegerTensorNames)
    %   Copyright 2024 The MathWorks, Inc. 
    if ismember(node.input{1}, IntegerTensorNames)
        op = '%s = fix(sqrt(sum(%s.^2, dims)));\n';
    else
        op = '%s = sqrt(sum(%s.^2, dims));\n';
    end
    nodeTranslation = translateReduceOp(this, nodeTranslation, node, IntegerTensorNames, op);
end
