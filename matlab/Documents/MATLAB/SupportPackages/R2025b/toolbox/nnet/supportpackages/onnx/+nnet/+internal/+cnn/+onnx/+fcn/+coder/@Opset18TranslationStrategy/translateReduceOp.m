function nodeTranslation = translateReduceOp(~, nodeTranslation, node, IntegerTensorNames, op)
    %   Copyright 2024-2025 The MathWorks, Inc. 

    % Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
    % To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
    AttributeTable = cell2table({
        "keepdims"              "INT"       true    1
        "noop_with_empty_axes"  "INT"       true    0
        });
    % Parse the attributes
    [keepdims, noop_with_empty_axes] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
    Y	= node.output{1};
    X	= node.input{1};
    dimsName = nnet.internal.cnn.onnx.fcn.uniqueCoderName('dims');
    reduceArg = nnet.internal.cnn.onnx.fcn.uniqueCoderName('xReduced'); 
    [VarY, NumDimY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
    [VarX, NumDimX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
    if numel(node.input) > 1 && ~isempty(node.input{2})
        % Axes is specified
        AxesName = node.input{2};
        [VarAxes, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, AxesName);
        
        baseReduceCommand = [
            sprintf('%s = %sprepareReduceArgs(%s, %s);\n', dimsName,...
            getPackageName(nodeTranslation.GraphTranslation), VarAxes, NumDimX),...
            sprintf(op, reduceArg, VarX, dimsName),...
            ];
        if keepdims            
            baseReduceCommand = [baseReduceCommand, sprintf('%s = %s;\n', VarY, reduceArg)];
            baseReduceCommand = [baseReduceCommand, sprintf('%s = %s;\n', NumDimY, NumDimX)];
            nodeTranslation.IncludedFunctionNames   = "prepareReduceArgs";
        else
            baseReduceCommand = [baseReduceCommand, sprintf('[%s, %s] = %sonnxSqueeze(%s, %s, %s);\n', ...
                VarY, NumDimY, getPackageName(nodeTranslation.GraphTranslation), reduceArg, VarAxes, NumDimX)];
            nodeTranslation.IncludedFunctionNames   = ["prepareReduceArgs", "onnxSqueeze"];
        end
        if  noop_with_empty_axes
            command = [
                sprintf('%% %s:\n', node.op_type),...
                sprintf('if isempty(%s)\n', VarAxes),...             
                sprintf('%s = %s;\n', VarY, VarX),... %do nothing
                sprintf('%s = %s;\n', NumDimY, NumDimX),...
                sprintf('else\n'),...
                baseReduceCommand,...
                sprintf('end;')
                ];
        else
            command = [
                sprintf('%% %s:\n', node.op_type),...
                baseReduceCommand];
        end
    else
        % Axes is empty
        if noop_with_empty_axes
            % Do nothing
            command = [
                sprintf('%% %s:\n', node.op_type),...                        
                sprintf('%s = %s;\n', VarY, VarX),...
                sprintf('%s = %s;\n', NumDimY, NumDimX)
            ];
        else
            % Reduce all axes
            command = [
                sprintf('%% %s:\n', node.op_type),...
                sprintf('%s = %sprepareReduceArgs([], %s);\n', dimsName, getPackageName(nodeTranslation.GraphTranslation), NumDimX),...
                sprintf(op, reduceArg,  VarX, dimsName),...
                ];                          
            if keepdims
                command = [command, sprintf('%s = %s;\n', VarY, reduceArg)];                     
                command = [command, sprintf('%s = %s;\n', NumDimY, NumDimX)];
                nodeTranslation.IncludedFunctionNames   = "prepareReduceArgs";
            else
                command = [command, sprintf('[%s, %s] = %sonnxSqueeze(%s, [], %s);\n', ...
                    VarY, NumDimY, getPackageName(nodeTranslation.GraphTranslation), reduceArg, NumDimX)];
                nodeTranslation.IncludedFunctionNames   = ["prepareReduceArgs", "onnxSqueeze"];       
            end                                      
        end
    end    
    nodeTranslation.MCode = string(command);    
    if ismember(X, IntegerTensorNames)
        nodeTranslation.IntegerOutputTensorNames = string(Y);
    end
end
