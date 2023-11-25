%% 
clc
clear all
close all


%% Problem definition
% n = 200;
% x1 = -1 + 2*rand(n,1); x2 = -1 + 2*rand(n,1); x3 = -1 + 2*rand(n,1); x4 = -1 + 2*rand(n,1);
% a = 10; b = 10; c = 10; d = 10; p = 1;
% y = a*(x1.^(p)) + b*x2.^p + c*x3.^p +d*x4.^p;
% data = [x1 x2 x3 x4 y];
n = 500;
%x1 = -1 + 2*rand(n,1); x2 = -1 + 2*rand(n,1); x3 = -1 + 2*rand(n,1); x4 = -1 + 2*rand(n,1);
%x1 = 1 + 9*rand(n,1); x2 = 1 + 9*rand(n,1); x3 = 1 + 9*rand(n,1); x4 = 1 + 9*rand(n,1);
x1 = -10 + 20*rand(n,1); x2 = -10 + 20*rand(n,1); x3 = -10 + 20*rand(n,1); x4 = -10 + 20*rand(n,1);

% x1 =x2.^2;
% x2 =x1.^2;
% x1 = 10*x2;
% x2 = 10*x1;
%disp(cos(1))
a = 1; b = 1; c = 10; d = 10; p = 2;
y = a*(x1.^(2)) + b*x2.^1 ;
%y = sin(x1) + x2 ;
data = [x1 x2 y];
disp(data);
%% Dataset
%sData_F = som_data_struct(data,'comp_names',{'x1','x2','y'});
%sData_F = som_normalize(sData_F,'range');
sData = som_data_struct(data,'comp_names',{'x1','x2','y'}); 
sData = som_normalize(sData,'range');
%% Initializing SOM Map Codebook Vectors (Linear Initialization)
%[sMap_F]= som_lininit(sData_F,'lattice','hexa','msize',[20,20]);
[sMap]= modifiedsom_lininit1(sData,'lattice','hexa','msize',[20,20]);
% [sMap]= modifiedsom_lininit(sData,'lattice','hexa');

%% Training SOM
%[sMap_F,sTrainF] = som_batchtrain(sMap_F,sData_F,'sample_order','ordered','trainlen',500,'radius_ini', 1.50, 'radius_fin',1.25);
% [sMap,sTrain] = modifiedsom_batchtrain(sMap,sData);
[sMap,sTrain] = modifiedsom_batchtrain(sMap,sData,'sample_order','ordered','trainlen',500,...
     'radius_ini', 1.0, 'radius_fin',0.9);
%% Denormalizing the data
%sMap_F = som_denormalize(sMap_F,sData_F); sData_F = som_denormalize(sData_F,'remove');
sMap=som_denormalize(sMap,sData);   sData=som_denormalize(sData,'remove');
%% Visualization of SOM results (U Matrix and Component Planes)
%figure(2) 
%som_show(sMap_F, 'umat','all','comp', 1:5);
figure(3) 
som_show(sMap, 'umat','all','comp', 'all');
annotation('textbox',[0.01,0.9,0.1,0.1],'String','y= x1^2 + x2^2','EdgeColor','none')

% annotation('textbox',[0.01,0.9,0.1,0.1],'String','y= x1^3 + 10*x2, x1 =x2.^2','EdgeColor','none')
% annotation('textbox',[0.01,0.9,0.1,0.1],'String','y= x1^3 + 10*x2, x2 =x1.^2','EdgeColor','none')
% annotation('textbox',[0.01,0.9,0.1,0.1],'String','y= x1^3 + 10*x2, x1 = 10*x2','EdgeColor','none')
% annotation('textbox',[0.01,0.9,0.1,0.1],'String','y= x1^3 + 10*x2, x2 = 10*x1','EdgeColor','none')
%% iSOM Grid in function space  
figure(4)
som_grid(sMap,'coord',sMap.codebook(:,[1 2 3]),'label',sMap.labels,'labelcolor','c','labelsize',10, 'marker','o','MarkerColor','k'...
    ,'MarkerSize',7,'linecolor', 'k');
hold on, scatter3(data(:,1),data(:,2),data(:,3),20,'ro','filled');
xlabel('x1')
ylabel('x2')
zlabel('y')
%% cSOM Grid in function space  
% figure(5)
% som_grid(sMap_F,'coord',sMap_F.codebook(:,[1 2 5]),'label',sMap.labels,'labelcolor','c','labelsize',10, 'marker','o','MarkerColor','k'...
%     ,'MarkerSize',7,'linecolor', 'k');
% hold on, scatter3(data(:,1),data(:,2),data(:,5),20,'ro','filled');
% xlabel('F1')
% ylabel('F2')
% zlabel('F3')



%% Assuming som_normalize function is available to normalize data
norm_data = som_normalize(sMap, 'range');

% Extracting the codebook vectors
v1 = norm_data.codebook(:,1);  % x1
v2 = norm_data.codebook(:,2);  % x2
v3 = norm_data.codebook(:,3);  % y (output)

% Compute slopes for each variable
x1_slope = calculate_slope(v1);
x2_slope = calculate_slope(v2);

% Project gradients of v3 (y) onto the slopes of each feature
[x1_pos, x1_neg] = projected_slope(v3, x1_slope);
[x2_pos, x2_neg] = projected_slope(v3, x2_slope);

% Aggregate all the results into a new variable
results.positive_correls = [x1_pos, x2_pos];
results.negative_correls = [x1_neg, x2_neg];

%% Plotting
features = {'x1', 'x2'};
figure;

subplot(2, 1, 1);
hold on;
barh(1:2, -results.negative_correls, 'r');
barh(1:2, results.positive_correls, 'g');
yticks(1:2); yticklabels(features);
grid on; hold off;

subplot(2, 1, 2);
barh(1:2, abs(results.positive_correls) + abs(results.negative_correls), 'b');
yticks(1:2); yticklabels(features);
grid on;





%%

function slope = calculate_slope(codebook_vec)
    data_ = reshape(codebook_vec, [20, 20])
    data_ = flipud(data_);
    [dZdx, dZdy] = gradient(data_);
    avg_dZdx = mean(dZdx(2:end-1, 2:end-1), 'all');
    avg_dZdy = mean(dZdy(2:end-1, 2:end-1), 'all');
    slope = [avg_dZdx, avg_dZdy];
end



function [positive_correl, negative_correl] = projected_slope(codebook_vec, input_slope)

    % Reshape and flip the codebook vector
    data_ = reshape(codebook_vec, [20, 20]);
    data_ = flipud(data_);
    
    % Calculate gradient
    [dZdx, dZdy] = gradient(data_);

    % Normalize the gradients
    grad_magnitudes = sqrt(dZdx.^2 + dZdy.^2);
    dZdx = dZdx ./ grad_magnitudes;
    dZdy = dZdy ./ grad_magnitudes;

    disp([dZdx, dZdy])

    % Normalize the input slope to get a unit vector
    magnitude = norm(input_slope); 
    unit_slope = input_slope / magnitude;
    
    % Initialize variables to hold the sum of vectors
    pos_sum = [0, 0];
    neg_sum = [0, 0];
    
    % Project dZdx and dZdy onto the input slope
    for i = 2:19
        for j = 2:19
            gradient_vec = [dZdx(i, j), dZdy(i, j)]; % make it a row vector
            dot_product = dot(gradient_vec, unit_slope);
            
            % Project gradient vector onto the unit slope
            projected_vec = (dot_product) * unit_slope;
            
            if dot_product > 0
                pos_sum = pos_sum + projected_vec;
            else
                neg_sum = neg_sum + projected_vec;
            end
        end
        
    end
    
    % Compute the magnitudes for vectors in each direction
    positive_correl = norm(pos_sum)/324;
    negative_correl = norm(neg_sum)/324;
    
end

% function y_slope = calculate_y_slope(output_codebook_vec, x_slope)
%     data_ = reshape(output_codebook_vec, [20, 20]);
%     data_ = flipud(data_);
%     [dZdx, dZdy] = gradient(data_);
%     [dZdx, dZdy] = reflectAboutPerpendicular(dZdx, dZdy, x_slope);
%     avg_dZdx = mean(dZdx(2:end-1, 2:end-1), 'all');
%     avg_dZdy = mean(dZdy(2:end-1, 2:end-1), 'all');
%     y_slope = [avg_dZdx, avg_dZdy];
%     a = y_slope';
%     b = x_slope';
%     proj_b_a = (dot(a, b) / dot(b, b)) * b;
%     y_slope = proj_b_a';
% end
% 
% % Define selective reflection function
% function [reflected_dx, reflected_dy] = reflectAboutPerpendicular(dx, dy, normal_vector)
%     % Define the perpendicular direction
%     n = [normal_vector(1); normal_vector(2)];
%     n = n / norm(n); % Normalize
%     
%     % Compute reflection matrix
%     R = eye(2) - 2 * (n * n');
%     
%     % Apply reflection matrix conditionally
%     for i = 1:size(dx, 1)
%         for j = 1:size(dy, 2)
%             if dot([dx(i,j); dy(i,j)], n) < 0
%                 result = R * [dx(i,j); dy(i,j)];
%                 reflected_dx(i,j) = result(1);
%                 reflected_dy(i,j) = result(2);
%             else
%                 reflected_dx(i,j) = dx(i,j);
%                 reflected_dy(i,j) = dy(i,j);
%             end
%         end
%     end
% end
% theta4 = 180*acos((x1'*x2)/(norm(x1)*norm(x2)))/pi;
% theta5 = 180*acos((x1'*y)/(norm(x1)*norm(y)))/pi;
% theta6 = 180*acos((y'*x2)/(norm(y)*norm(x2)))/pi;