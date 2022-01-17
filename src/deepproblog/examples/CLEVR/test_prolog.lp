
:- dynamic query_size/2, relate_left/2, relate_right/2, relate_front/2, relate_behind/2, filter_large/2, filter_small/2,
    filter_gray/2, filter_red/2, filter_blue/2, filter_green/2, filter_brown/2, filter_purple/2, filter_cyan/2,
    filter_yellow/2, filter_metal/2, filter_rubber/2, filter_sphere/2, filter_cylinder/2, filter_cube/2, and/3, or/3,
    same_size/2, same_color/2, same_shape/2, same_material/2,
    query_material/2, query_shape/2, query_color/2, query_size/2,
    equal_integer/3, count/2, exist/2, less_than/3, greater_than/3, 
    equal_size/3, equal_color/3, equal_material/3, equal_shape/3.
:- style_check(-singleton).
:-discontiguous(obj/10).
scene(0,0).

filter_small(1,1).
filter_small(6,6).
filter_yellow(2,2).
filter_metal(3,3).
filter_metal(7,7).
same_shape(5,5).
query_color(9,9).
end(9).

obj(T+1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- scene(T,0), obj(0,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2).
obj(T+1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- unique(T,T2), obj(T2,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2).
obj(T+1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- relate_left(T,T2), obj(0,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), obj(T2,ID2,SIZE2,COLOR2,MATERIAL2,SHAPE2,X12,Y12,X22,Y22), ID=\=ID2, X1<X12.
obj(T+1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- relate_right(T,T2), obj(0,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), obj(T2,ID2,SIZE2,COLOR2,MATERIAL2,SHAPE2,X12,Y12,X22,Y22), ID=\=ID2, X1>=X12.
obj(T+1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- relate_front(T,T2), obj(0,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), obj(T2,ID2,SIZE2,COLOR2,MATERIAL2,SHAPE2,X12,Y12,X22,Y22), ID=\=ID2, Y2>Y22.
obj(T+1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- relate_behind(T,T2), obj(0,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), obj(T2,ID2,SIZE2,COLOR2,MATERIAL2,SHAPE2,X12,Y12,X22,Y22), ID=\=ID2, Y2=<Y22.
obj(T+1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- filter_large(T,T2), obj(T2,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), SIZE=large.
obj(T+1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- filter_small(T,T2), obj(T2,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), SIZE=small.
obj(T+1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- filter_gray(T,T2), obj(T2,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), COLOR=gray.
obj(T+1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- filter_red(T,T2), obj(T2,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), COLOR=red.
obj(T+1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- filter_blue(T,T2), obj(T2,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), COLOR=blue.
obj(T+1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- filter_green(T,T2), obj(T2,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), COLOR=green.
obj(T+1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- filter_brown(T,T2), obj(T2,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), COLOR=brown.
obj(T+1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- filter_purple(T,T2), obj(T2,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), COLOR=purple.
obj(T+1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- filter_cyan(T,T2), obj(T2,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), COLOR=cyan.
obj(T+1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- filter_yellow(T,T2), obj(T2,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), COLOR=yellow.
obj(T+1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- filter_metal(T,T2), obj(T2,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), MATERIAL=metal.
obj(T+1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- filter_rubber(T,T2), obj(T2,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), MATERIAL=rubber.
obj(T+1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- filter_sphere(T,T2), obj(T2,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), SHAPE=sphere.
obj(T+1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- filter_cylinder(T,T2), obj(T2,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), SHAPE=cylinder.
obj(T+1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- filter_cube(T,T2), obj(T2,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), SHAPE=cube.
obj(T+1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- and(T,T2,T22), obj(T2,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), obj(T22,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2).
obj(0,1,small,green,metal,cylinder,113,155,154,205).
obj(0,2,small,yellow,rubber,cylinder,152,182,195,238).
obj(0,3,small,cyan,rubber,cylinder,263,199,309,259).
obj(0,4,large,purple,metal,sphere,343,130,420,205).
obj(0,5,small,cyan,metal,cube,182,74,218,110).
obj(0,6,large,gray,metal,cylinder,47,84,123,171).
obj(0,7,large,blue,rubber,cylinder,382,84,457,171).
obj(0,8,small,yellow,metal,cube,281,93,319,110).
obj(0,9,large,blue,metal,cube,265,95,343,191).
obj(0,1,small,green,metal,cylinder,113,155,154,205).
obj(0,2,small,yellow,rubber,cylinder,152,182,195,238).
obj(0,3,small,cyan,rubber,cylinder,263,199,309,259).
obj(0,4,large,purple,metal,sphere,343,130,420,205).
obj(0,5,small,cyan,metal,cube,182,74,218,110).
obj(0,6,large,gray,metal,cylinder,47,84,123,171).
obj(0,7,large,blue,rubber,cylinder,382,84,457,171).
obj(0,8,small,yellow,metal,cube,281,93,319,110).
obj(0,9,large,blue,metal,cube,265,95,343,191).

unique(8,8).
unique(4,4).
:- unique(T,T2), obj(T2,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), obj(T2,ID2,SIZE2,COLOR2,MATERIAL2,SHAPE2,X12,Y12,X22,Y22), ID=\=ID2.


% Count rule
int(T,V) :- count(T,T2), count(ID) is V, obj(T2,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2).

% Exist rule


% Filtering rules

% Query functions
size(T,SIZE) :- query_size(T,T2), obj(T2,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2).
color(T,COLOR) :- query_color(T,T2), obj(T2,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2).
material(T,MATERIAL) :- query_material(T,T2), obj(T2,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2).
shape(T,SHAPE) :- query_shape(T,T2), obj(T2,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2).

% Logical operators

or(T,T2) :- or(T,T2,T22).
or(T,T22) :- or(T,T2,T22).
obj(T+1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- or(T,T2), obj(T2,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2).

% Same-attribute relations
obj(T+1,ID2,SIZE2,COLOR2,MATERIAL2,SHAPE2,X12,Y12,X22,Y22) :- same_size(T,T2), obj(T2,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), obj(0,ID2,SIZE2,COLOR2,MATERIAL2,SHAPE2,X12,Y12,X22,Y22), ID=\=ID2, SIZE=SIZE2.
obj(T+1,ID2,SIZE2,COLOR2,MATERIAL2,SHAPE2,X12,Y12,X22,Y22) :- same_color(T,T2), obj(T2,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), obj(0,ID2,SIZE2,COLOR2,MATERIAL2,SHAPE2,X12,Y12,X22,Y22), ID=\=ID2, COLOR=COLOR2.
obj(T+1,ID2,SIZE2,COLOR2,MATERIAL2,SHAPE2,X12,Y12,X22,Y22) :- same_material(T,T2), obj(T2,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), obj(0,ID2,SIZE2,COLOR2,MATERIAL2,SHAPE2,X12,Y12,X22,Y22), ID=\=ID2, MATERIAL=MATERIAL2.
obj(T+1,ID2,SIZE2,COLOR2,MATERIAL2,SHAPE2,X12,Y12,X22,Y22) :- same_shape(T,T2), obj(T2,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), obj(0,ID2,SIZE2,COLOR2,MATERIAL2,SHAPE2,X12,Y12,X22,Y22), ID=\=ID2, SHAPE=SHAPE2.

% Integer comparison
bool(T,true) :- equal_integer(T,T2,T22), int(T2,V2), int(T22,V22), V2=V22.
bool(T,false) :- equal_integer(T,T2,T22), \+  bool(T,true).
bool(T,true) :- exist(T,T2), obj(T2,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2).
bool(T,false) :- exist(T,T2), \+  bool(T,true).
bool(T,true) :- less_than(T,T2,T22), int(T2,V2), int(T22,V22), V2<V22.
bool(T,false) :- less_than(T,T2,T22), \+  bool(T,true).
bool(T,true) :- greater_than(T,T2,T22), int(T2,V2), int(T22,V22), V2>V22.
bool(T,false) :- greater_than(T,T2,T22), \+  bool(T,true).
% Attribute comparison
bool(T,true) :- equal_size(T,T2,T22), size(T2,V2), size(T22,V22), V2=V22.
bool(T,false) :- equal_size(T,T2,T22), \+  bool(T,true).
bool(T,true) :- equal_color(T,T2,T22), color(T2,V2), color(T22,V22), V2=V22.
bool(T,false) :- equal_color(T,T2,T22), \+  bool(T,true).
bool(T,true) :- equal_material(T,T2,T22), material(T2,V2), material(T22,V22), V2=V22.
bool(T,false) :- equal_material(T,T2,T22), \+  bool(T,true).
bool(T,true) :- equal_shape(T,T2,T22), shape(T2,V2), shape(T22,V22), V2=V22.
bool(T,false) :- equal_shape(T,T2,T22), \+  bool(T,true).

% Derive answer (T must equal the last point in time)
ans(V) :- end(T), size(T,V).
ans(V) :- end(T), color(T,V).
ans(V) :- end(T), material(T,V).
ans(V) :- end(T), shape(T,V).
ans(V) :- end(T), bool(T,V).
ans(V) :- end(T), int(T,V).

% obj(0,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- label(0,img,ID,obj(ID,SHAPE,SIZE,COLOR,MATERIAL,X1,Y1,X2,Y2)).obj(0,0,small,blue,metal,cylinder,212,96,241,134).


