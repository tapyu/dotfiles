classdef ConverterForONNXElementwiseAffineLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    
    % Copyright 2018-2021 The MathWorks, Inc.
    
    methods
        function this = ConverterForONNXElementwiseAffineLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*
            % Implementation of Y=X.*A+B: C=Mul(X,A) followed by Add(C,B).
            % Two initialiers added to model.
            
            existingNodeNames   = {nodeProto.name};
            [onnxName, ~]       = legalizeNNTName(this, this.NNTLayer.Name);
            needMul             = ~all(this.NNTLayer.Scale(:) == 1);
            needAdd             = ~all(this.NNTLayer.Offset(:) == 0) || ~needMul;   % Ensure we always generate a node.
            inputTensorNames    = mapTensorNames(this, this.InputLayerNames, TensorNameMap);
            inputTensorLayout   = TensorLayoutMap(inputTensorNames{1});
            
            switch inputTensorLayout
                case 'nchw'
                    permIdx = [3 1 2];
                case 'nchwd'
                    permIdx = [4 1 2 3];
                case {'snc', '1nc'}
                    permIdx = [1 2 3]; 
                case 'nc'
                    permIdx = [2 1];
                case 'snchw'
                    permIdx = [3 1 2];
                case 'snchwd'
                    permIdx = [4 1 2 3];
                otherwise
                    error(message('nnet_cnn_onnx:onnx:UnexpectedONNXInputLayout', this.NNTLayer.Name));
            end
            
            % (1) Mul
            if needMul
                MulNodeName       = [onnxName, '_Mul'];
                MulNodeName       = makeUniqueName(existingNodeNames, MulNodeName);
                existingNodeNames = [existingNodeNames, {MulNodeName}];
                MulConstName      = [MulNodeName '_B'];
                mulNode           = NodeProto;
                mulNode.op_type   = 'Mul';
                mulNode.name      = MulNodeName;
                mulNode.input     = {inputTensorNames{1}, MulConstName};
                mulNode.output    = {MulNodeName};
                if this.OpsetVersion < 7
                    % Include broadcast and axis:
                    mulNode.attribute  = [...
                        makeAttributeProto('broadcast', 'INT', 1),...
                        makeAttributeProto('axis',      'INT', 1),...
                        ];
                end
                % Initializer. 
                t1              = TensorProto;
                t1.name         = MulConstName;
                t1.data_type    = TensorProto_DataType.FLOAT;
                Scale           = this.NNTLayer.Scale;
                Scale           = permute(Scale, permIdx);                % CHW, CHWD, or C cm
                t1.raw_data     = rawData(single(Scale));                 % CHW, CHWD, or C rm
                t1.dims         = dimVector(size(Scale), numel(permIdx)); % dims can be CHW, CHWD, or C

            else
                mulNode = [];
                t1      = [];
            end
            
            % (2) Add
            if needAdd
                if needMul
                    AddInputName = MulNodeName;
                else
                    AddInputName = inputTensorNames{1};
                end
                AddNodeName  	  = [onnxName, '_Add'];
                AddNodeName       = makeUniqueName(existingNodeNames, AddNodeName);
                AddConstName      = [AddNodeName, '_B'];
                addNode           = NodeProto;
                addNode.op_type   = 'Add';
                addNode.name      = AddNodeName;
                addNode.input     = {AddInputName, AddConstName};
                addNode.output    = {AddNodeName};
                if this.OpsetVersion < 7
                    % Include broadcast and axis:
                    addNode.attribute  = [...
                        makeAttributeProto('broadcast', 'INT', 1),...
                        makeAttributeProto('axis',      'INT', 1),...
                        ];
                end
                % Initializer. 
                t2              = TensorProto;
                t2.name         = AddConstName;
                t2.data_type    = TensorProto_DataType.FLOAT;
                Offset          = this.NNTLayer.Offset;
                Offset          = permute(Offset, permIdx);               % CHW, CHWD, or C cm
                t2.raw_data     = rawData(single(Offset));             	  % CHW, CHWD, or C rm
                t2.dims         = dimVector(size(Offset),numel(permIdx)); % dims can be CHW, CHWD, or C

            else
                addNode = [];
                t2      = [];
            end
            
            nodeProto               = [nodeProto mulNode addNode];
            parameterInitializers   = [t1 t2];
            networkInputs        	= [];
            networkOutputs        	= [];
            
            % Update maps
            outputTensorName                  = nodeProto(end).name;
            TensorNameMap(this.NNTLayer.Name) = outputTensorName;
            TensorLayoutMap(outputTensorName) = inputTensorLayout;
            
            % Update network if current layer is danglingLayer
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
        end
    end
end
