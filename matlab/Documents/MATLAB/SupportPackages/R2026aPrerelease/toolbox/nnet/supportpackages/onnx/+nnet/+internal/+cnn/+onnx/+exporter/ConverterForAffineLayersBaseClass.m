classdef ConverterForAffineLayersBaseClass < nnet.internal.cnn.onnx.exporter.NNTLayerConverter
    % A DLT layer that does Y=aX+b with scalar (a,b). If b~=0, it is
    % exported as Mul(a) followed by Add(b). If b==0. it is just Mul(a).
    
    % Copyright 2018-2025 The MathWorks, Inc.
    
    
    properties
        a
        b
    end
    
    methods
        function this = ConverterForAffineLayersBaseClass(layerAnalyzer)
            this@nnet.internal.cnn.onnx.exporter.NNTLayerConverter(layerAnalyzer);
        end
        
        function this = set.a(this, a)
            assert(isnumeric(a) && isscalar(a), ...
                message('nnet_cnn_onnx:onnx:InvalidElementwiseAffineScaleOrOffset'));
            this.a = a;
        end
        
        function this = set.b(this, b)
            assert(isnumeric(b) && isscalar(b), ...
                message('nnet_cnn_onnx:onnx:InvalidElementwiseAffineScaleOrOffset'));
            this.b = b;
        end
        
        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*
            % Implementation of Y=aX+b: Mul(a) followed by Add(b).
                        
            inputTensorNames    = mapTensorNames(this, this.InputLayerNames(:)', TensorNameMap);
            inputTensorLayout   = TensorLayoutMap(inputTensorNames{1});

            [onnxName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            existingNodeNames       = {nodeProto.name};
            
            % (1) Mul
            MulNodeName             = [onnxName, '_Mul'];
            MulNodeName             = makeUniqueName(existingNodeNames, MulNodeName);
            MulBName                = [MulNodeName '_B'];
            [newNodes(1), ~] = createNodeProto(this, 'SISO','Mul', MulNodeName, ...
                mapTensorNames(this, {this.InputLayerNames{1}, MulBName}, TensorNameMap), {MulNodeName});
            if this.OpsetVersion < 7
                newNodes(1).attribute = makeAttributeProto('broadcast', 'INT', 1);
            end
            % Initializer for B
            t1              = TensorProto;
            t1.name         = MulBName;
            t1.data_type    = TensorProto_DataType.FLOAT;
            t1.raw_data     = rawData(single(this.a));
            t1.dims         = dimVector(1,1);
            parameterInitializers(1) = t1;
            
            % (2) If b~=0, also generate an Add operator: Adds two inputs A
            % and B. We'll set A to be the output of the Mul and set B to
            % be the bias. We'll give B an initializer.
            if this.b ~= 0
                AddNodeName            = [onnxName, '_Add'];
                AddNodeName            = makeUniqueName([existingNodeNames, {MulNodeName}], AddNodeName);
                AddBName               = [AddNodeName, '_B'];
                [newNodes(2), ~] = createNodeProto(this, 'SISO','Add', AddNodeName,...
                    {MulNodeName, AddBName}, {AddNodeName});
                if this.OpsetVersion < 7
                    newNodes(2).attribute = makeAttributeProto('broadcast',	'INT', 1);
                end
                % Initializer for B
                t2              = TensorProto;
                t2.name         = AddBName;
                t2.data_type    = TensorProto_DataType.FLOAT;
                t2.raw_data     = rawData(single(this.b));
                t2.dims      	= dimVector(1,1);
                
                parameterInitializers(2) = t2;
            end
            
            nodeProto               = [nodeProto newNodes];
            networkInputs        	= [];
            networkOutputs        	= [];
              
            % Update maps
            outputTensorName                    = newNodes(end).name;  % Output from last node
            TensorNameMap(this.NNTLayer.Name)   = outputTensorName;
            outputTensorLayout                  = inputTensorLayout;
            TensorLayoutMap(outputTensorName)   = outputTensorLayout;
            
            % Update network if current layer is danglingLayer
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
        end
    end
end
