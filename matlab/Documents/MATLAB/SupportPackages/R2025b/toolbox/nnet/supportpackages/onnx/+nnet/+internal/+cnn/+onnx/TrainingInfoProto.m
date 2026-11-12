classdef TrainingInfoProto
    % Language binding class for the TrainingInfoProto protobuf message    
    % optional GraphProto initialization = 1;
    % optional GraphProto algorithm = 2;
    % repeated StringStringEntryProto initialization_binding = 3;
    % repeated StringStringEntryProto update_binding = 4;
    
    %   Copyright 2021 The MathWorks, Inc.
    
    properties
        initialization
        algorithm
        initialization_binding
        update_binding
    end
    
    methods
        function this = TrainingInfoProto(varargin) 
            import nnet.internal.cnn.onnx.*
            %Add Spkg bin path to system path
            nnet.internal.cnn.onnx.util.addSpkgBinPath();
            if nargin > 0
                Ptr = varargin{1};
                % Get raw properties
                PropertyCell = onnxmex(int32(FuncName.EdecodeTrainingInfoProto), Ptr);
                [this.initialization, this.algorithm,...
                    this.initialization_binding, this.update_binding] = PropertyCell{:};
                % Call constructors on Proto objects
                if ~isempty(this.initialization)
                    this.initialization = GraphProto(this.initialization);
                end
                if ~isempty(this.algorithm)
                    this.algorithm = GraphProto(this.algorithm);
                end        
                if ~isempty(this.initialization_binding)
                    this.initialization_binding = arrayfun(@StringStringEntryProto, this.initialization_binding);
                end
                if ~isempty(this.update_binding)
                    this.update_binding = arrayfun(@StringStringEntryProto, this.update_binding);
                end
            end
        end
        
        function encodeTrainingInfoProto(this, CPtr)
             % Recursively fill the CPtr from 'this'.
            import nnet.internal.cnn.onnx.*
            PropertyCell = {this.initialization, this.algorithm, ...
                 this.initialization_binding, this.update_binding};
            PtrCell = onnxmex(int32(FuncName.EencodeTrainingInfoProto), CPtr, PropertyCell);
            % Fill pointer objects
            arrayfun(@encodeGraphProto, this.initialization, PtrCell{1});
            arrayfun(@encodeGraphProto, this.algorithm, PtrCell{2});  
            arrayfun(@encodeStringStringEntryProto, this.initialization_binding, PtrCell{3});
            arrayfun(@encodeStringStringEntryProto, this.update_binding, PtrCell{4});            
        end
    end
end
