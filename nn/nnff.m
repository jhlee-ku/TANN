function nn = nnff(nn, x, y)
% NNFF performs a feedforward pass
% nn = nnff(nn, x, y) returns an neural network structure with updated
% layer activations, error and loss (nn.a, nn.e and nn.L)

    n = nn.n;       % n : # of layers
    m = size(x, 1); % m : # of training samples
    
    x = [ones(m,1) x];  % add bias
    nn.a{1} = x;        % input is activation 1

    % Feedforward pass
    for i = 2 : n-1     % output (n layer) is calcuated later
        switch nn.activation_function 
            case 'sigm'
                nn.a{i} = sigm(nn.a{i - 1} * nn.W{i - 1}');
            case 'tanh_opt'
                nn.a{i} = tanh_opt(nn.a{i - 1} * nn.W{i - 1}');
            case 'tanh'
                nn.a{i} = tanh(nn.a{i - 1} * nn.W{i - 1}');
            case 'relu'		% Rectified linear unit (ReLU); By JH Lee Jan. 25, 2017
                nn.a{i} = max(0.1,sign(nn.a{i - 1} * nn.W{i - 1}')) .* (nn.a{i - 1} * nn.W{i - 1}'); % leaky ReLU
        end
        
        % Dropout
        if(nn.dropoutFraction > 0)
            if(nn.testing)
                nn.a{i} = nn.a{i}.*(1 - nn.dropoutFraction);
            else
                nn.dropOutMask{i} = (rand(size(nn.a{i}))>nn.dropoutFraction);
                nn.a{i} = nn.a{i}.*nn.dropOutMask{i};
            end
        end
        
        % Sparsity
        if(nn.nonSparsityPenalty>0)
             nn.p{i} = 0.99 * nn.p{i} + 0.01 * mean(nn.a{i}, 1); 
%             nn.p{i} = mean(nn.a{i}, 1); % nn.p : average value in batches
        end
        
        % Add the bias term
        nn.a{i} = [ones(m,1) nn.a{i}];
    end
    
    switch nn.output 
        case 'sigm'     % sigmoid : sigmoid(activation*weight)
            nn.a{n} = sigm(nn.a{n - 1} * nn.W{n - 1}'); 
        case 'tanh'     % tanh: tanh(activation*weight)
            nn.a{n} = tanh(nn.a{n - 1} * nn.W{n - 1}'); 
        case 'linear'   % linear : activation*weight
            nn.a{n} = nn.a{n - 1} * nn.W{n - 1}'; 
        case 'softmax'  % softmax : exp(activation*weight - max)/sum(exp)            
            nn.a{n} = nn.a{n - 1} * nn.W{n - 1}';   
            nn.a{n} = exp(bsxfun(@minus, nn.a{n}, max(nn.a{n},[],2)));  
            nn.a{n} = bsxfun(@rdivide, nn.a{n}, sum(nn.a{n}, 2));
    end

    % Error and Loss
    nn.e = y - nn.a{n};
    
    % Loss function is different by output function.
    % classification (layer>3) : cross entropy + softmax. why cross-entropy?
    % regression : leasat sqaure error + linear unit.
    switch nn.output
        case {'sigm', 'linear','tanh'}     % sigmoid, linear : MSE (mean squared error)
            nn.l = 1/2 * sum(sum(nn.e .^ 2)) / m;   
        case 'softmax'              % softmax : Cross-entropy. Sum_batch(Sum_class(prob(if correct,1)*prob(class)))). cf) http://newsight.tistory.com/category.
            nn.l = -sum(sum(y .* log(nn.a{n}))) / m;    % y. class label. log. acitvation is exponential. 
    end
	 %keyboard;
%	nn.abser= sum(sum(abs(nn.e)))/m;
	nn.varerr= var(nn.e(:))/var(y(:));
end
