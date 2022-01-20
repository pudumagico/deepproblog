:- use_module(library(aggregate)).
% :- dynamic unique/2.
unique(-1,-1).
% :- dynamic relate_left/2.
relate_left(-1,-1).
% :- dynamic relate_right/2.
relate_right(-1,-1).
% :- dynamic relate_front/2.
relate_front(-1,-1).
% :- dynamic relate_behind/2.
relate_behind(-1,-1).
% :- dynamic count/2.
count(-1,-1).
% :- dynamic exist/2.
exist(-1,-1).
% :- dynamic filter_large/2.
filter_large(-1,-1).
% :- dynamic filter_small/2.
filter_small(-1,-1).
% :- dynamic filter_gray/2.
filter_gray(-1,-1).
% :- dynamic filter_red/2.
filter_red(-1,-1).
% :- dynamic filter_blue/2.
filter_blue(-1,-1).
% :- dynamic filter_green/2.
filter_green(-1,-1).
% :- dynamic filter_brown/2.
filter_brown(-1,-1).
% :- dynamic filter_purple/2.
filter_purple(-1,-1).
% :- dynamic filter_cyan/2.
filter_cyan(-1,-1).
% :- dynamic filter_yellow/2.
filter_yellow(-1,-1).
% :- dynamic filter_metal/2.
filter_metal(-1,-1).
% :- dynamic filter_rubber/2.
filter_rubber(-1,-1).
% :- dynamic filter_sphere/2.
filter_sphere(-1,-1).
% :- dynamic filter_cylinder/2.
filter_cylinder(-1,-1).
% :- dynamic filter_cube/2.
filter_cube(-1,-1).
% :- dynamic query_size/2.
query_size(-1,-1).
% :- dynamic query_color/2.
query_color(-1,-1).
% :- dynamic query_material/2.
query_material(-1,-1).
% :- dynamic query_shape/2.
query_shape(-1,-1).
% :- dynamic and/3.
and(-1,-1,-1).
% :- dynamic or/2.
or(-1,-1).
% :- dynamic or/3.
or(-1,-1,-1).
% :- dynamic same_size/2.
same_size(-1,-1).
% :- dynamic same_color/2.
same_color(-1,-1).
% :- dynamic same_material/2.
same_material(-1,-1).
% :- dynamic same_shape/2.
same_shape(-1,-1).
% :- dynamic equal_integer/3.
equal_integer(-1,-1,-1).
% :- dynamic less_than/3.
less_than(-1,-1,-1).
% :- dynamic greater_than/3.
greater_than(-1,-1,-1).
% :- dynamic equal_size/3.
equal_size(-1,-1,-1).
% :- dynamic equal_color/3.
equal_color(-1,-1,-1).
% :- dynamic equal_material/3.
equal_material(-1,-1,-1).
% :- dynamic equal_shape/3.
equal_shape(-1,-1,-1).
% :- dynamic size/2.
size(-1,-1).
% :- dynamic color/2.
color(-1,-1).
% :- dynamic material/2.
material(-1,-1).
% :- dynamic shape/2.
shape(-1,-1).
% :- dynamic bool/2.
bool(-1,-1).
% :- dynamic int/2.
int(-1,-1).
% :- dynamic int/3.
int(-1,-1).
% :- dynamic label/4.
label(-1,-1,-1,-1).
% :- dynamic end/1.
end(-2).
% :- dynamic scene/2.
scene(-1,-1).

% Scene rule
obj(Z,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- scene(T,0), Z is T+1, obj(0,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2).

% Unique rule/constraint
% obj(Z,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- unique(T,T1), Z is T+1, obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2).
%:- unique(T,T1), obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), obj(T1,ID1,SIZE1,COLOR1,MATERIAL1,SHAPE1,X11,Y11,X21,Y21), ID=\=ID1.
obj(Z,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- unique(T,T1), Z is T+1, obj(T,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), obj(T1,ID1,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), ID \= ID1,!, fail.
obj(Z,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- unique(T,T1), Z is T+1, obj(T,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2).


% Relate rules
obj(Z,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- relate_left(T,T1), Z is T+1, obj(0,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), obj(T1,ID1,SIZE1,COLOR1,MATERIAL1,SHAPE1,X11,Y11,X21,Y21), ID=\=ID1, X1<X11.
obj(Z,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- relate_right(T,T1), Z is T+1, obj(0,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), obj(T1,ID1,SIZE1,COLOR1,MATERIAL1,SHAPE1,X11,Y11,X21,Y21), ID=\=ID1, X1>=X11.
obj(Z,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- relate_front(T,T1), Z is T+1, obj(0,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), obj(T1,ID1,SIZE1,COLOR1,MATERIAL1,SHAPE1,X11,Y11,X21,Y21), ID=\=ID1, Y2>Y21.
obj(Z,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- relate_behind(T,T1), Z is T+1, obj(0,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), obj(T1,ID1,SIZE1,COLOR1,MATERIAL1,SHAPE1,X11,Y11,X21,Y21), ID=\=ID1, Y2=<Y21.

% Count rule
int(T,N,ID,1) :- count(T,T1), obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2).
int(T, sum<X>) :- count(T,T1), int(T,N,ID,X).

% Exist rule
bool(T,true) :- exist(T,T1), obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2).
bool(T,false) :- exist(T,T1), \+ bool(T,true).

% Filtering rules
obj(Z,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- filter_large(T,T1), Z is T+1, obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), SIZE=large.
obj(Z,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- filter_small(T,T1), Z is T+1, obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), SIZE=small.
obj(Z,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- filter_gray(T,T1), Z is T+1, obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), COLOR=gray.
obj(Z,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- filter_red(T,T1), Z is T+1, obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), COLOR=red.
obj(Z,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- filter_blue(T,T1), Z is T+1, obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), COLOR=blue.
obj(Z,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- filter_green(T,T1), Z is T+1, obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), COLOR=green.
obj(Z,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- filter_brown(T,T1), Z is T+1, obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), COLOR=brown.
obj(Z,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- filter_purple(T,T1), Z is T+1, obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), COLOR=purple.
obj(Z,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- filter_cyan(T,T1), Z is T+1, obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), COLOR=cyan.
obj(Z,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- filter_yellow(T,T1), Z is T+1, obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), COLOR=yellow.
obj(Z,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- filter_metal(T,T1), Z is T+1, obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), MATERIAL=metal.
obj(Z,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- filter_rubber(T,T1), Z is T+1, obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), MATERIAL=rubber.
obj(Z,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- filter_sphere(T,T1), Z is T+1, obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), SHAPE=sphere.
obj(Z,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- filter_cylinder(T,T1), Z is T+1, obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), SHAPE=cylinder.
obj(Z,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- filter_cube(T,T1), Z is T+1, obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), SHAPE=cube.

% Query functions
size(T,SIZE) :- query_size(T,T1), obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2).
color(T,COLOR) :- query_color(T,T1), obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2).
material(T,MATERIAL) :- query_material(T,T1), obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2).
shape(T,SHAPE) :- query_shape(T,T1), obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2).

% Logical operators
obj(Z,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- and(T,T1,T2), Z is T+1, obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), obj(T2,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2).

or(T,T1) :- or(T,T1,T2).
or(T,T2) :- or(T,T1,T2).
obj(Z,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- or(T,T1), Z is T+1, obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2).

% Same-attribute relations
obj(Z,ID1,SIZE1,COLOR1,MATERIAL1,SHAPE1,X11,Y11,X21,Y21) :- same_size(T,T1), Z is T+1, obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), obj(0,ID1,SIZE1,COLOR1,MATERIAL1,SHAPE1,X11,Y11,X21,Y21), ID=\=ID1, SIZE=SIZE1.
obj(Z,ID1,SIZE1,COLOR1,MATERIAL1,SHAPE1,X11,Y11,X21,Y21) :- same_color(T,T1), Z is T+1, obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), obj(0,ID1,SIZE1,COLOR1,MATERIAL1,SHAPE1,X11,Y11,X21,Y21), ID=\=ID1, COLOR=COLOR1.
obj(Z,ID1,SIZE1,COLOR1,MATERIAL1,SHAPE1,X11,Y11,X21,Y21) :- same_material(T,T1), Z is T+1, obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), obj(0,ID1,SIZE1,COLOR1,MATERIAL1,SHAPE1,X11,Y11,X21,Y21), ID=\=ID1, MATERIAL=MATERIAL1.
obj(Z,ID1,SIZE1,COLOR1,MATERIAL1,SHAPE1,X11,Y11,X21,Y21) :- same_shape(T,T1), Z is T+1, obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), obj(0,ID1,SIZE1,COLOR1,MATERIAL1,SHAPE1,X11,Y11,X21,Y21), ID=\=ID1, SHAPE=SHAPE1.

% Integer comparison
bool(T,true) :- equal_integer(T,T1,T2), int(T1,V1), int(T2,V2), V1=V2.
bool(T,false) :- equal_integer(T,T1,T2), \+ bool(T,true).

bool(T,true) :- less_than(T,T1,T2), int(T1,V1), int(T2,V2), V1<V2.
bool(T,false) :- less_than(T,T1,T2), \+ bool(T,true).

bool(T,true) :- greater_than(T,T1,T2), int(T1,V1), int(T2,V2), V1>V2.
bool(T,false) :- greater_than(T,T1,T2), \+ bool(T,true).

% Attribute comparison
bool(T,true) :- equal_size(T,T1,T2), size(T1,V1), size(T2,V2), V1=V2.
bool(T,false) :- equal_size(T,T1,T2), \+ bool(T,true).

bool(T,true) :- equal_color(T,T1,T2), color(T1,V1), color(T2,V2), V1=V2.
bool(T,false) :- equal_color(T,T1,T2), \+ bool(T,true).

bool(T,true) :- equal_material(T,T1,T2), material(T1,V1), material(T2,V2), V1=V2.
bool(T,false) :- equal_material(T,T1,T2), \+ bool(T,true).

bool(T,true) :- equal_shape(T,T1,T2), shape(T1,V1), shape(T2,V2), V1=V2.
bool(T,false) :- equal_shape(T,T1,T2), \+ bool(T,true).

% Derive answer (T must equal the last point in time)
ans(V) :- end(T), size(T,V).
ans(V) :- end(T), color(T,V).
ans(V) :- end(T), material(T,V).
ans(V) :- end(T), shape(T,V).
ans(V) :- end(T), bool(T,V).
ans(V) :- end(T), int(T,V).

obj(0,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- label(0,img,ID,obj(ID,SHAPE,SIZE,COLOR,MATERIAL,X1,Y1,X2,Y2)).
