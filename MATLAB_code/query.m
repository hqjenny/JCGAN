%% SAVE THE OBJECT WITH ITS MASK AND SOURCE IMAGE&FOLDER
boundingbox = [];
centers = [];
obj_Dobj = [];
this_Dobj = [];
for i = 1:length(select_name)
    this_name = select_name{i};
    [Dobj, ~] = LMquery(D, 'object.name', this_name);
    for ii = 1:length(Dobj)
        [this_boundingbox, this_centers] = LMobjectboundingbox(Dobj(ii).annotation);
        [mask, ~] = LMobjectmask(Dobj(ii).annotation, HOMEIMAGES);
        imgsize = size(mask,1)*size(mask,2);
        this_area = (this_boundingbox(:,3)-this_boundingbox(:,1)).*(this_boundingbox(:,4)-this_boundingbox(:,2));
        this_mask = this_area >= imgsize * 0.02;
        this_mask_one = sum(this_mask);
        boundingbox = this_boundingbox(this_mask,:);
        centers = this_centers(this_mask,:);
        
        this_object = Dobj(ii).annotation.object;
        this_object = this_object(this_mask);
        for iii = 1:this_mask_one
            % this_Dobj = rmfield(Dobj(ii).annotation,{'source', 'imagesize'});
            this_Dobj.annotation = Dobj(ii).annotation;
            this_Dobj.annotation.object = this_object(iii);
            this_Dobj.annotation.object.boundingbox = boundingbox(iii,:);
            this_Dobj.annotation.object.centers = centers(iii,:);
            obj_Dobj = [obj_Dobj; this_Dobj];
        end
    end
end

save('obj_Dobj','obj_Dobj');

% LMobjectnormalizedcrop

%% visualize each object
[Dcar, j] = LMquery(D, 'object.name', 'car');

[x,y] = LMobjectpolygon(this_Dobj.annotation, 1);
figure
plot(x{1}, y{1}, 'r')
axis('ij')

%% query on folder name
%j: indices to images that fit the query. Note Dout is not D(j). D(j)
%    contains all the objects in the original struct

[Df] = LMquery(D, 'folder', 'spatial_envelope_256x256_static_8outdoorcategories');

%% query on shape matching
[Dq, ind_shape] = LMquerypolygon(D, 'car', 1, 1, .5);

%% look for objects by occlusion:
[D_occlude,j_occlude] = LMquery(D, 'object.occluded', 'no');

[D_occlude_car, ~] = LMquery(D_occlude, 'object.name', 'car');
LMdbshowscenes(D_occlude_car, HOMEIMAGES);

%%
obj = obj_Dobj_all(77355).annotation;
[mask, class] = LMobjectmask(obj, HOMEIMAGES);
imshow(colorSegments(mask))
title(obj.object.name)

[boundingbox centers] = LMobjectboundingbox(obj); % [xmin, ymin, xmax, ymax]
boundingbox

%% Get the top 100 list of object names and counts

% [names, counts] = LMobjectnames(D);
% let's train only on car, people and tree

[sorted_count, ind] = sort(counts, 'descend');
sorted_name = names(ind(1:100));
sorted_ct = counts(ind(1:100));
save('sorted_name', 'sorted_name')
save('sorted_ct', 'sorted_ct')

%%
% first 300 categories
USED_ind = [2, 3, 5, 9, 15, 17, 23, 28, 31, 42, 47, 63, 66, 81, 82, 83, ...
    110, 111, 119, 126, 127, 138, 145, 147, 160, 166, 177, 180, 181, 197, ...
    202, 210, 232, 252, 256, 263, 281, 294];
select_name = sorted_name(USED_ind);

%% VISUALIZE THE SELECTED OBJECTS

[Dobj, j] = LMquery(D, 'object.name', 'plant pot');

LMdbshowscenes(Dobj(1), HOMEIMAGES)

