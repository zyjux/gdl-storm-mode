ЂЕ'
ЉС
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
Ы
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

Ѓ
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
М
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
В
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
>
Minimum
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
f
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx" 
Tidxtype0:
2
	
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
n
	ReverseV2
tensor"T
axis"Tidx
output"T"
Tidxtype0:
2	"
Ttype:
2	

l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
Ѕ
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И®
@
StaticRegexFullMatch	
input

output
"
patternstring
ч
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
y
TensorScatterUpdate
tensor"T
indices"Tindices
updates"T
output"T"	
Ttype"
Tindicestype:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.9.12v2.9.0-18-gd8ce9f9c3018сч#
А
Nadam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameNadam/dense_1/bias/v
y
(Nadam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_1/bias/v*
_output_shapes
:*
dtype0
И
Nadam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameNadam/dense_1/kernel/v
Б
*Nadam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_1/kernel/v*
_output_shapes

: *
dtype0
|
Nadam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameNadam/dense/bias/v
u
&Nadam/dense/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense/bias/v*
_output_shapes
: *
dtype0
Е
Nadam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А *%
shared_nameNadam/dense/kernel/v
~
(Nadam/dense/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense/kernel/v*
_output_shapes
:	А *
dtype0
Ч
Nadam/rot_equiv_conv2d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*0
shared_name!Nadam/rot_equiv_conv2d_4/bias/v
Р
3Nadam/rot_equiv_conv2d_4/bias/v/Read/ReadVariableOpReadVariableOpNadam/rot_equiv_conv2d_4/bias/v*
_output_shapes	
:А*
dtype0
І
!Nadam/rot_equiv_conv2d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*2
shared_name#!Nadam/rot_equiv_conv2d_4/kernel/v
†
5Nadam/rot_equiv_conv2d_4/kernel/v/Read/ReadVariableOpReadVariableOp!Nadam/rot_equiv_conv2d_4/kernel/v*'
_output_shapes
:@А*
dtype0
Ц
Nadam/rot_equiv_conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Nadam/rot_equiv_conv2d_3/bias/v
П
3Nadam/rot_equiv_conv2d_3/bias/v/Read/ReadVariableOpReadVariableOpNadam/rot_equiv_conv2d_3/bias/v*
_output_shapes
:@*
dtype0
¶
!Nadam/rot_equiv_conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*2
shared_name#!Nadam/rot_equiv_conv2d_3/kernel/v
Я
5Nadam/rot_equiv_conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOp!Nadam/rot_equiv_conv2d_3/kernel/v*&
_output_shapes
:@@*
dtype0
Ц
Nadam/rot_equiv_conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Nadam/rot_equiv_conv2d_2/bias/v
П
3Nadam/rot_equiv_conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpNadam/rot_equiv_conv2d_2/bias/v*
_output_shapes
:@*
dtype0
¶
!Nadam/rot_equiv_conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*2
shared_name#!Nadam/rot_equiv_conv2d_2/kernel/v
Я
5Nadam/rot_equiv_conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOp!Nadam/rot_equiv_conv2d_2/kernel/v*&
_output_shapes
: @*
dtype0
Ц
Nadam/rot_equiv_conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Nadam/rot_equiv_conv2d_1/bias/v
П
3Nadam/rot_equiv_conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpNadam/rot_equiv_conv2d_1/bias/v*
_output_shapes
: *
dtype0
¶
!Nadam/rot_equiv_conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *2
shared_name#!Nadam/rot_equiv_conv2d_1/kernel/v
Я
5Nadam/rot_equiv_conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOp!Nadam/rot_equiv_conv2d_1/kernel/v*&
_output_shapes
:  *
dtype0
Т
Nadam/rot_equiv_conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nameNadam/rot_equiv_conv2d/bias/v
Л
1Nadam/rot_equiv_conv2d/bias/v/Read/ReadVariableOpReadVariableOpNadam/rot_equiv_conv2d/bias/v*
_output_shapes
: *
dtype0
Ґ
Nadam/rot_equiv_conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Nadam/rot_equiv_conv2d/kernel/v
Ы
3Nadam/rot_equiv_conv2d/kernel/v/Read/ReadVariableOpReadVariableOpNadam/rot_equiv_conv2d/kernel/v*&
_output_shapes
: *
dtype0
А
Nadam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameNadam/dense_1/bias/m
y
(Nadam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_1/bias/m*
_output_shapes
:*
dtype0
И
Nadam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameNadam/dense_1/kernel/m
Б
*Nadam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_1/kernel/m*
_output_shapes

: *
dtype0
|
Nadam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameNadam/dense/bias/m
u
&Nadam/dense/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense/bias/m*
_output_shapes
: *
dtype0
Е
Nadam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А *%
shared_nameNadam/dense/kernel/m
~
(Nadam/dense/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense/kernel/m*
_output_shapes
:	А *
dtype0
Ч
Nadam/rot_equiv_conv2d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*0
shared_name!Nadam/rot_equiv_conv2d_4/bias/m
Р
3Nadam/rot_equiv_conv2d_4/bias/m/Read/ReadVariableOpReadVariableOpNadam/rot_equiv_conv2d_4/bias/m*
_output_shapes	
:А*
dtype0
І
!Nadam/rot_equiv_conv2d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*2
shared_name#!Nadam/rot_equiv_conv2d_4/kernel/m
†
5Nadam/rot_equiv_conv2d_4/kernel/m/Read/ReadVariableOpReadVariableOp!Nadam/rot_equiv_conv2d_4/kernel/m*'
_output_shapes
:@А*
dtype0
Ц
Nadam/rot_equiv_conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Nadam/rot_equiv_conv2d_3/bias/m
П
3Nadam/rot_equiv_conv2d_3/bias/m/Read/ReadVariableOpReadVariableOpNadam/rot_equiv_conv2d_3/bias/m*
_output_shapes
:@*
dtype0
¶
!Nadam/rot_equiv_conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*2
shared_name#!Nadam/rot_equiv_conv2d_3/kernel/m
Я
5Nadam/rot_equiv_conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOp!Nadam/rot_equiv_conv2d_3/kernel/m*&
_output_shapes
:@@*
dtype0
Ц
Nadam/rot_equiv_conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Nadam/rot_equiv_conv2d_2/bias/m
П
3Nadam/rot_equiv_conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpNadam/rot_equiv_conv2d_2/bias/m*
_output_shapes
:@*
dtype0
¶
!Nadam/rot_equiv_conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*2
shared_name#!Nadam/rot_equiv_conv2d_2/kernel/m
Я
5Nadam/rot_equiv_conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOp!Nadam/rot_equiv_conv2d_2/kernel/m*&
_output_shapes
: @*
dtype0
Ц
Nadam/rot_equiv_conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Nadam/rot_equiv_conv2d_1/bias/m
П
3Nadam/rot_equiv_conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpNadam/rot_equiv_conv2d_1/bias/m*
_output_shapes
: *
dtype0
¶
!Nadam/rot_equiv_conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *2
shared_name#!Nadam/rot_equiv_conv2d_1/kernel/m
Я
5Nadam/rot_equiv_conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOp!Nadam/rot_equiv_conv2d_1/kernel/m*&
_output_shapes
:  *
dtype0
Т
Nadam/rot_equiv_conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nameNadam/rot_equiv_conv2d/bias/m
Л
1Nadam/rot_equiv_conv2d/bias/m/Read/ReadVariableOpReadVariableOpNadam/rot_equiv_conv2d/bias/m*
_output_shapes
: *
dtype0
Ґ
Nadam/rot_equiv_conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Nadam/rot_equiv_conv2d/kernel/m
Ы
3Nadam/rot_equiv_conv2d/kernel/m/Read/ReadVariableOpReadVariableOpNadam/rot_equiv_conv2d/kernel/m*&
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
|
Nadam/momentum_cacheVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameNadam/momentum_cache
u
(Nadam/momentum_cache/Read/ReadVariableOpReadVariableOpNadam/momentum_cache*
_output_shapes
: *
dtype0
z
Nadam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameNadam/learning_rate
s
'Nadam/learning_rate/Read/ReadVariableOpReadVariableOpNadam/learning_rate*
_output_shapes
: *
dtype0
j
Nadam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/decay
c
Nadam/decay/Read/ReadVariableOpReadVariableOpNadam/decay*
_output_shapes
: *
dtype0
l
Nadam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/beta_2
e
 Nadam/beta_2/Read/ReadVariableOpReadVariableOpNadam/beta_2*
_output_shapes
: *
dtype0
l
Nadam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/beta_1
e
 Nadam/beta_1/Read/ReadVariableOpReadVariableOpNadam/beta_1*
_output_shapes
: *
dtype0
h

Nadam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
Nadam/iter
a
Nadam/iter/Read/ReadVariableOpReadVariableOp
Nadam/iter*
_output_shapes
: *
dtype0	
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

: *
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
: *
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А *
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	А *
dtype0
З
rot_equiv_conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*(
shared_namerot_equiv_conv2d_4/bias
А
+rot_equiv_conv2d_4/bias/Read/ReadVariableOpReadVariableOprot_equiv_conv2d_4/bias*
_output_shapes	
:А*
dtype0
Ч
rot_equiv_conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А**
shared_namerot_equiv_conv2d_4/kernel
Р
-rot_equiv_conv2d_4/kernel/Read/ReadVariableOpReadVariableOprot_equiv_conv2d_4/kernel*'
_output_shapes
:@А*
dtype0
Ж
rot_equiv_conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_namerot_equiv_conv2d_3/bias

+rot_equiv_conv2d_3/bias/Read/ReadVariableOpReadVariableOprot_equiv_conv2d_3/bias*
_output_shapes
:@*
dtype0
Ц
rot_equiv_conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@**
shared_namerot_equiv_conv2d_3/kernel
П
-rot_equiv_conv2d_3/kernel/Read/ReadVariableOpReadVariableOprot_equiv_conv2d_3/kernel*&
_output_shapes
:@@*
dtype0
Ж
rot_equiv_conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_namerot_equiv_conv2d_2/bias

+rot_equiv_conv2d_2/bias/Read/ReadVariableOpReadVariableOprot_equiv_conv2d_2/bias*
_output_shapes
:@*
dtype0
Ц
rot_equiv_conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @**
shared_namerot_equiv_conv2d_2/kernel
П
-rot_equiv_conv2d_2/kernel/Read/ReadVariableOpReadVariableOprot_equiv_conv2d_2/kernel*&
_output_shapes
: @*
dtype0
Ж
rot_equiv_conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_namerot_equiv_conv2d_1/bias

+rot_equiv_conv2d_1/bias/Read/ReadVariableOpReadVariableOprot_equiv_conv2d_1/bias*
_output_shapes
: *
dtype0
Ц
rot_equiv_conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  **
shared_namerot_equiv_conv2d_1/kernel
П
-rot_equiv_conv2d_1/kernel/Read/ReadVariableOpReadVariableOprot_equiv_conv2d_1/kernel*&
_output_shapes
:  *
dtype0
В
rot_equiv_conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_namerot_equiv_conv2d/bias
{
)rot_equiv_conv2d/bias/Read/ReadVariableOpReadVariableOprot_equiv_conv2d/bias*
_output_shapes
: *
dtype0
Т
rot_equiv_conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_namerot_equiv_conv2d/kernel
Л
+rot_equiv_conv2d/kernel/Read/ReadVariableOpReadVariableOprot_equiv_conv2d/kernel*&
_output_shapes
: *
dtype0

NoOpNoOp
ВИ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ЉЗ
value±ЗB≠З B•З
Ѓ
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer-10
layer_with_weights-5
layer-11
layer_with_weights-6
layer-12
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
µ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
	filt_base
bias*
Ш
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses
%pool* 
µ
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses

,kernel
,	filt_base
-bias*
Ш
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses
4pool* 
µ
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses

;kernel
;	filt_base
<bias*
Ш
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses
Cpool* 
µ
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses

Jkernel
J	filt_base
Kbias*
Ш
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses
Rpool* 
µ
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses

Ykernel
Y	filt_base
Zbias*
О
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses* 
О
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses* 
¶
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses

mkernel
nbias*
¶
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses

ukernel
vbias*
j
0
1
,2
-3
;4
<5
J6
K7
Y8
Z9
m10
n11
u12
v13*
j
0
1
,2
-3
;4
<5
J6
K7
Y8
Z9
m10
n11
u12
v13*
* 
∞
wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
|trace_0
}trace_1
~trace_2
trace_3* 
:
Аtrace_0
Бtrace_1
Вtrace_2
Гtrace_3* 
* 
ц
	Дiter
Еbeta_1
Жbeta_2

Зdecay
Иlearning_rate
Йmomentum_cachem•m¶,mІ-m®;m©<m™JmЂKmђYm≠ZmЃmmѓnm∞um±vm≤v≥vі,vµ-vґ;vЈ<vЄJvєKvЇYvїZvЉmvљnvЊuvњvvј*

Кserving_default* 

0
1*

0
1*
* 
Ш
Лnon_trainable_variables
Мlayers
Нmetrics
 Оlayer_regularization_losses
Пlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Рtrace_0* 

Сtrace_0* 
ga
VARIABLE_VALUErot_equiv_conv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUErot_equiv_conv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ц
Тnon_trainable_variables
Уlayers
Фmetrics
 Хlayer_regularization_losses
Цlayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses* 

Чtrace_0* 

Шtrace_0* 
Ф
Щ	variables
Ъtrainable_variables
Ыregularization_losses
Ь	keras_api
Э__call__
+Ю&call_and_return_all_conditional_losses* 

,0
-1*

,0
-1*
* 
Ш
Яnon_trainable_variables
†layers
°metrics
 Ґlayer_regularization_losses
£layer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses*

§trace_0* 

•trace_0* 
ic
VARIABLE_VALUErot_equiv_conv2d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUErot_equiv_conv2d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ц
¶non_trainable_variables
Іlayers
®metrics
 ©layer_regularization_losses
™layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses* 

Ђtrace_0* 

ђtrace_0* 
Ф
≠	variables
Ѓtrainable_variables
ѓregularization_losses
∞	keras_api
±__call__
+≤&call_and_return_all_conditional_losses* 

;0
<1*

;0
<1*
* 
Ш
≥non_trainable_variables
іlayers
µmetrics
 ґlayer_regularization_losses
Јlayer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses*

Єtrace_0* 

єtrace_0* 
ic
VARIABLE_VALUErot_equiv_conv2d_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUErot_equiv_conv2d_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ц
Їnon_trainable_variables
їlayers
Љmetrics
 љlayer_regularization_losses
Њlayer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses* 

њtrace_0* 

јtrace_0* 
Ф
Ѕ	variables
¬trainable_variables
√regularization_losses
ƒ	keras_api
≈__call__
+∆&call_and_return_all_conditional_losses* 

J0
K1*

J0
K1*
* 
Ш
«non_trainable_variables
»layers
…metrics
  layer_regularization_losses
Ћlayer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses*

ћtrace_0* 

Ќtrace_0* 
ic
VARIABLE_VALUErot_equiv_conv2d_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUErot_equiv_conv2d_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ц
ќnon_trainable_variables
ѕlayers
–metrics
 —layer_regularization_losses
“layer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses* 

”trace_0* 

‘trace_0* 
Ф
’	variables
÷trainable_variables
„regularization_losses
Ў	keras_api
ў__call__
+Џ&call_and_return_all_conditional_losses* 

Y0
Z1*

Y0
Z1*
* 
Ш
џnon_trainable_variables
№layers
Ёmetrics
 ёlayer_regularization_losses
яlayer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses*

аtrace_0* 

бtrace_0* 
ic
VARIABLE_VALUErot_equiv_conv2d_4/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUErot_equiv_conv2d_4/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ц
вnon_trainable_variables
гlayers
дmetrics
 еlayer_regularization_losses
жlayer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses* 

зtrace_0* 

иtrace_0* 
* 
* 
* 
Ц
йnon_trainable_variables
кlayers
лmetrics
 мlayer_regularization_losses
нlayer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses* 

оtrace_0* 

пtrace_0* 

m0
n1*

m0
n1*
* 
Ш
рnon_trainable_variables
сlayers
тmetrics
 уlayer_regularization_losses
фlayer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses*

хtrace_0* 

цtrace_0* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

u0
v1*

u0
v1*
* 
Ш
чnon_trainable_variables
шlayers
щmetrics
 ъlayer_regularization_losses
ыlayer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses*

ьtrace_0* 

эtrace_0* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
b
0
1
2
3
4
5
6
7
	8

9
10
11
12*

ю0
€1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
MG
VARIABLE_VALUE
Nadam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUENadam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUENadam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUENadam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUENadam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUENadam/momentum_cache3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
%0* 
* 
* 
* 
* 
* 
* 
* 
* 
Ь
Аnon_trainable_variables
Бlayers
Вmetrics
 Гlayer_regularization_losses
Дlayer_metrics
Щ	variables
Ъtrainable_variables
Ыregularization_losses
Э__call__
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses* 

Еtrace_0* 

Жtrace_0* 
* 
* 
* 
* 
* 
* 
* 
* 
	
40* 
* 
* 
* 
* 
* 
* 
* 
* 
Ь
Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
≠	variables
Ѓtrainable_variables
ѓregularization_losses
±__call__
+≤&call_and_return_all_conditional_losses
'≤"call_and_return_conditional_losses* 

Мtrace_0* 

Нtrace_0* 
* 
* 
* 
* 
* 
* 
* 
* 
	
C0* 
* 
* 
* 
* 
* 
* 
* 
* 
Ь
Оnon_trainable_variables
Пlayers
Рmetrics
 Сlayer_regularization_losses
Тlayer_metrics
Ѕ	variables
¬trainable_variables
√regularization_losses
≈__call__
+∆&call_and_return_all_conditional_losses
'∆"call_and_return_conditional_losses* 

Уtrace_0* 

Фtrace_0* 
* 
* 
* 
* 
* 
* 
* 
* 
	
R0* 
* 
* 
* 
* 
* 
* 
* 
* 
Ь
Хnon_trainable_variables
Цlayers
Чmetrics
 Шlayer_regularization_losses
Щlayer_metrics
’	variables
÷trainable_variables
„regularization_losses
ў__call__
+Џ&call_and_return_all_conditional_losses
'Џ"call_and_return_conditional_losses* 

Ъtrace_0* 

Ыtrace_0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
Ь	variables
Э	keras_api

Юtotal

Яcount*
M
†	variables
°	keras_api

Ґtotal

£count
§
_fn_kwargs*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

Ю0
Я1*

Ь	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

Ґ0
£1*

†	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
МЕ
VARIABLE_VALUENadam/rot_equiv_conv2d/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ИБ
VARIABLE_VALUENadam/rot_equiv_conv2d/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ОЗ
VARIABLE_VALUE!Nadam/rot_equiv_conv2d_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
КГ
VARIABLE_VALUENadam/rot_equiv_conv2d_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ОЗ
VARIABLE_VALUE!Nadam/rot_equiv_conv2d_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
КГ
VARIABLE_VALUENadam/rot_equiv_conv2d_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ОЗ
VARIABLE_VALUE!Nadam/rot_equiv_conv2d_3/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
КГ
VARIABLE_VALUENadam/rot_equiv_conv2d_3/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ОЗ
VARIABLE_VALUE!Nadam/rot_equiv_conv2d_4/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
КГ
VARIABLE_VALUENadam/rot_equiv_conv2d_4/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUENadam/dense/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUENadam/dense/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUENadam/dense_1/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUENadam/dense_1/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
МЕ
VARIABLE_VALUENadam/rot_equiv_conv2d/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ИБ
VARIABLE_VALUENadam/rot_equiv_conv2d/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ОЗ
VARIABLE_VALUE!Nadam/rot_equiv_conv2d_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
КГ
VARIABLE_VALUENadam/rot_equiv_conv2d_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ОЗ
VARIABLE_VALUE!Nadam/rot_equiv_conv2d_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
КГ
VARIABLE_VALUENadam/rot_equiv_conv2d_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ОЗ
VARIABLE_VALUE!Nadam/rot_equiv_conv2d_3/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
КГ
VARIABLE_VALUENadam/rot_equiv_conv2d_3/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ОЗ
VARIABLE_VALUE!Nadam/rot_equiv_conv2d_4/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
КГ
VARIABLE_VALUENadam/rot_equiv_conv2d_4/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUENadam/dense/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUENadam/dense/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUENadam/dense_1/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUENadam/dense_1/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Э
&serving_default_rot_equiv_conv2d_inputPlaceholder*1
_output_shapes
:€€€€€€€€€РР*
dtype0*&
shape:€€€€€€€€€РР
Ш
StatefulPartitionedCallStatefulPartitionedCall&serving_default_rot_equiv_conv2d_inputrot_equiv_conv2d/kernelrot_equiv_conv2d/biasrot_equiv_conv2d_1/kernelrot_equiv_conv2d_1/biasrot_equiv_conv2d_2/kernelrot_equiv_conv2d_2/biasrot_equiv_conv2d_3/kernelrot_equiv_conv2d_3/biasrot_equiv_conv2d_4/kernelrot_equiv_conv2d_4/biasdense/kernel
dense/biasdense_1/kerneldense_1/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *-
f(R&
$__inference_signature_wrapper_625958
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
к
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename+rot_equiv_conv2d/kernel/Read/ReadVariableOp)rot_equiv_conv2d/bias/Read/ReadVariableOp-rot_equiv_conv2d_1/kernel/Read/ReadVariableOp+rot_equiv_conv2d_1/bias/Read/ReadVariableOp-rot_equiv_conv2d_2/kernel/Read/ReadVariableOp+rot_equiv_conv2d_2/bias/Read/ReadVariableOp-rot_equiv_conv2d_3/kernel/Read/ReadVariableOp+rot_equiv_conv2d_3/bias/Read/ReadVariableOp-rot_equiv_conv2d_4/kernel/Read/ReadVariableOp+rot_equiv_conv2d_4/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpNadam/iter/Read/ReadVariableOp Nadam/beta_1/Read/ReadVariableOp Nadam/beta_2/Read/ReadVariableOpNadam/decay/Read/ReadVariableOp'Nadam/learning_rate/Read/ReadVariableOp(Nadam/momentum_cache/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp3Nadam/rot_equiv_conv2d/kernel/m/Read/ReadVariableOp1Nadam/rot_equiv_conv2d/bias/m/Read/ReadVariableOp5Nadam/rot_equiv_conv2d_1/kernel/m/Read/ReadVariableOp3Nadam/rot_equiv_conv2d_1/bias/m/Read/ReadVariableOp5Nadam/rot_equiv_conv2d_2/kernel/m/Read/ReadVariableOp3Nadam/rot_equiv_conv2d_2/bias/m/Read/ReadVariableOp5Nadam/rot_equiv_conv2d_3/kernel/m/Read/ReadVariableOp3Nadam/rot_equiv_conv2d_3/bias/m/Read/ReadVariableOp5Nadam/rot_equiv_conv2d_4/kernel/m/Read/ReadVariableOp3Nadam/rot_equiv_conv2d_4/bias/m/Read/ReadVariableOp(Nadam/dense/kernel/m/Read/ReadVariableOp&Nadam/dense/bias/m/Read/ReadVariableOp*Nadam/dense_1/kernel/m/Read/ReadVariableOp(Nadam/dense_1/bias/m/Read/ReadVariableOp3Nadam/rot_equiv_conv2d/kernel/v/Read/ReadVariableOp1Nadam/rot_equiv_conv2d/bias/v/Read/ReadVariableOp5Nadam/rot_equiv_conv2d_1/kernel/v/Read/ReadVariableOp3Nadam/rot_equiv_conv2d_1/bias/v/Read/ReadVariableOp5Nadam/rot_equiv_conv2d_2/kernel/v/Read/ReadVariableOp3Nadam/rot_equiv_conv2d_2/bias/v/Read/ReadVariableOp5Nadam/rot_equiv_conv2d_3/kernel/v/Read/ReadVariableOp3Nadam/rot_equiv_conv2d_3/bias/v/Read/ReadVariableOp5Nadam/rot_equiv_conv2d_4/kernel/v/Read/ReadVariableOp3Nadam/rot_equiv_conv2d_4/bias/v/Read/ReadVariableOp(Nadam/dense/kernel/v/Read/ReadVariableOp&Nadam/dense/bias/v/Read/ReadVariableOp*Nadam/dense_1/kernel/v/Read/ReadVariableOp(Nadam/dense_1/bias/v/Read/ReadVariableOpConst*A
Tin:
826	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *(
f#R!
__inference__traced_save_628148
’
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamerot_equiv_conv2d/kernelrot_equiv_conv2d/biasrot_equiv_conv2d_1/kernelrot_equiv_conv2d_1/biasrot_equiv_conv2d_2/kernelrot_equiv_conv2d_2/biasrot_equiv_conv2d_3/kernelrot_equiv_conv2d_3/biasrot_equiv_conv2d_4/kernelrot_equiv_conv2d_4/biasdense/kernel
dense/biasdense_1/kerneldense_1/bias
Nadam/iterNadam/beta_1Nadam/beta_2Nadam/decayNadam/learning_rateNadam/momentum_cachetotal_1count_1totalcountNadam/rot_equiv_conv2d/kernel/mNadam/rot_equiv_conv2d/bias/m!Nadam/rot_equiv_conv2d_1/kernel/mNadam/rot_equiv_conv2d_1/bias/m!Nadam/rot_equiv_conv2d_2/kernel/mNadam/rot_equiv_conv2d_2/bias/m!Nadam/rot_equiv_conv2d_3/kernel/mNadam/rot_equiv_conv2d_3/bias/m!Nadam/rot_equiv_conv2d_4/kernel/mNadam/rot_equiv_conv2d_4/bias/mNadam/dense/kernel/mNadam/dense/bias/mNadam/dense_1/kernel/mNadam/dense_1/bias/mNadam/rot_equiv_conv2d/kernel/vNadam/rot_equiv_conv2d/bias/v!Nadam/rot_equiv_conv2d_1/kernel/vNadam/rot_equiv_conv2d_1/bias/v!Nadam/rot_equiv_conv2d_2/kernel/vNadam/rot_equiv_conv2d_2/bias/v!Nadam/rot_equiv_conv2d_3/kernel/vNadam/rot_equiv_conv2d_3/bias/v!Nadam/rot_equiv_conv2d_4/kernel/vNadam/rot_equiv_conv2d_4/bias/vNadam/dense/kernel/vNadam/dense/bias/vNadam/dense_1/kernel/vNadam/dense_1/bias/v*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *+
f&R$
"__inference__traced_restore_628314√Ў!
ЯЁ
т
!__inference__wrapped_model_624800
rot_equiv_conv2d_inputY
?sequential_rot_equiv_conv2d_convolution_readvariableop_resource: I
;sequential_rot_equiv_conv2d_biasadd_readvariableop_resource: [
Asequential_rot_equiv_conv2d_1_convolution_readvariableop_resource:  K
=sequential_rot_equiv_conv2d_1_biasadd_readvariableop_resource: [
Asequential_rot_equiv_conv2d_2_convolution_readvariableop_resource: @K
=sequential_rot_equiv_conv2d_2_biasadd_readvariableop_resource:@[
Asequential_rot_equiv_conv2d_3_convolution_readvariableop_resource:@@K
=sequential_rot_equiv_conv2d_3_biasadd_readvariableop_resource:@\
Asequential_rot_equiv_conv2d_4_convolution_readvariableop_resource:@АL
=sequential_rot_equiv_conv2d_4_biasadd_readvariableop_resource:	АB
/sequential_dense_matmul_readvariableop_resource:	А >
0sequential_dense_biasadd_readvariableop_resource: C
1sequential_dense_1_matmul_readvariableop_resource: @
2sequential_dense_1_biasadd_readvariableop_resource:
identityИҐ'sequential/dense/BiasAdd/ReadVariableOpҐ&sequential/dense/MatMul/ReadVariableOpҐ)sequential/dense_1/BiasAdd/ReadVariableOpҐ(sequential/dense_1/MatMul/ReadVariableOpҐ2sequential/rot_equiv_conv2d/BiasAdd/ReadVariableOpҐ*sequential/rot_equiv_conv2d/ReadVariableOpҐ,sequential/rot_equiv_conv2d/ReadVariableOp_1Ґ,sequential/rot_equiv_conv2d/ReadVariableOp_2Ґ6sequential/rot_equiv_conv2d/convolution/ReadVariableOpҐ4sequential/rot_equiv_conv2d_1/BiasAdd/ReadVariableOpҐ8sequential/rot_equiv_conv2d_1/convolution/ReadVariableOpҐ:sequential/rot_equiv_conv2d_1/convolution_1/ReadVariableOpҐ:sequential/rot_equiv_conv2d_1/convolution_2/ReadVariableOpҐ:sequential/rot_equiv_conv2d_1/convolution_3/ReadVariableOpҐ4sequential/rot_equiv_conv2d_2/BiasAdd/ReadVariableOpҐ8sequential/rot_equiv_conv2d_2/convolution/ReadVariableOpҐ:sequential/rot_equiv_conv2d_2/convolution_1/ReadVariableOpҐ:sequential/rot_equiv_conv2d_2/convolution_2/ReadVariableOpҐ:sequential/rot_equiv_conv2d_2/convolution_3/ReadVariableOpҐ4sequential/rot_equiv_conv2d_3/BiasAdd/ReadVariableOpҐ8sequential/rot_equiv_conv2d_3/convolution/ReadVariableOpҐ:sequential/rot_equiv_conv2d_3/convolution_1/ReadVariableOpҐ:sequential/rot_equiv_conv2d_3/convolution_2/ReadVariableOpҐ:sequential/rot_equiv_conv2d_3/convolution_3/ReadVariableOpҐ4sequential/rot_equiv_conv2d_4/BiasAdd/ReadVariableOpҐ8sequential/rot_equiv_conv2d_4/convolution/ReadVariableOpҐ:sequential/rot_equiv_conv2d_4/convolution_1/ReadVariableOpҐ:sequential/rot_equiv_conv2d_4/convolution_2/ReadVariableOpҐ:sequential/rot_equiv_conv2d_4/convolution_3/ReadVariableOpЊ
6sequential/rot_equiv_conv2d/convolution/ReadVariableOpReadVariableOp?sequential_rot_equiv_conv2d_convolution_readvariableop_resource*&
_output_shapes
: *
dtype0о
'sequential/rot_equiv_conv2d/convolutionConv2Drot_equiv_conv2d_input>sequential/rot_equiv_conv2d/convolution/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ОО *
paddingVALID*
strides
Ј
/sequential/rot_equiv_conv2d/Rank/ReadVariableOpReadVariableOp?sequential_rot_equiv_conv2d_convolution_readvariableop_resource*&
_output_shapes
: *
dtype0b
 sequential/rot_equiv_conv2d/RankConst*
_output_shapes
: *
dtype0*
value	B :i
'sequential/rot_equiv_conv2d/range/startConst*
_output_shapes
: *
dtype0*
value	B : i
'sequential/rot_equiv_conv2d/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :’
!sequential/rot_equiv_conv2d/rangeRange0sequential/rot_equiv_conv2d/range/start:output:0)sequential/rot_equiv_conv2d/Rank:output:00sequential/rot_equiv_conv2d/range/delta:output:0*
_output_shapes
:Р
7sequential/rot_equiv_conv2d/TensorScatterUpdate/indicesConst*
_output_shapes

:*
dtype0*!
valueB"       И
7sequential/rot_equiv_conv2d/TensorScatterUpdate/updatesConst*
_output_shapes
:*
dtype0*
valueB"       Ђ
/sequential/rot_equiv_conv2d/TensorScatterUpdateTensorScatterUpdate*sequential/rot_equiv_conv2d/range:output:0@sequential/rot_equiv_conv2d/TensorScatterUpdate/indices:output:0@sequential/rot_equiv_conv2d/TensorScatterUpdate/updates:output:0*
T0*
Tindices0*
_output_shapes
:≤
*sequential/rot_equiv_conv2d/ReadVariableOpReadVariableOp?sequential_rot_equiv_conv2d_convolution_readvariableop_resource*&
_output_shapes
: *
dtype0d
"sequential/rot_equiv_conv2d/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :t
*sequential/rot_equiv_conv2d/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:ћ
%sequential/rot_equiv_conv2d/ReverseV2	ReverseV22sequential/rot_equiv_conv2d/ReadVariableOp:value:03sequential/rot_equiv_conv2d/ReverseV2/axis:output:0*
T0*&
_output_shapes
: Ќ
%sequential/rot_equiv_conv2d/transpose	Transpose.sequential/rot_equiv_conv2d/ReverseV2:output:08sequential/rot_equiv_conv2d/TensorScatterUpdate:output:0*
T0*&
_output_shapes
: џ
)sequential/rot_equiv_conv2d/convolution_1Conv2Drot_equiv_conv2d_input)sequential/rot_equiv_conv2d/transpose:y:0*
T0*1
_output_shapes
:€€€€€€€€€ОО *
paddingVALID*
strides
d
"sequential/rot_equiv_conv2d/Rank_2Const*
_output_shapes
: *
dtype0*
value	B :k
)sequential/rot_equiv_conv2d/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : k
)sequential/rot_equiv_conv2d/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :Ё
#sequential/rot_equiv_conv2d/range_1Range2sequential/rot_equiv_conv2d/range_1/start:output:0+sequential/rot_equiv_conv2d/Rank_2:output:02sequential/rot_equiv_conv2d/range_1/delta:output:0*
_output_shapes
:Т
9sequential/rot_equiv_conv2d/TensorScatterUpdate_1/indicesConst*
_output_shapes

:*
dtype0*!
valueB"      К
9sequential/rot_equiv_conv2d/TensorScatterUpdate_1/updatesConst*
_output_shapes
:*
dtype0*
valueB"      ≥
1sequential/rot_equiv_conv2d/TensorScatterUpdate_1TensorScatterUpdate,sequential/rot_equiv_conv2d/range_1:output:0Bsequential/rot_equiv_conv2d/TensorScatterUpdate_1/indices:output:0Bsequential/rot_equiv_conv2d/TensorScatterUpdate_1/updates:output:0*
T0*
Tindices0*
_output_shapes
:а
'sequential/rot_equiv_conv2d/transpose_1	Transpose2sequential/rot_equiv_conv2d/convolution_1:output:0:sequential/rot_equiv_conv2d/TensorScatterUpdate_1:output:0*
T0*1
_output_shapes
:€€€€€€€€€ОО d
"sequential/rot_equiv_conv2d/Rank_3Const*
_output_shapes
: *
dtype0*
value	B :v
,sequential/rot_equiv_conv2d/ReverseV2_1/axisConst*
_output_shapes
:*
dtype0*
valueB:‘
'sequential/rot_equiv_conv2d/ReverseV2_1	ReverseV2+sequential/rot_equiv_conv2d/transpose_1:y:05sequential/rot_equiv_conv2d/ReverseV2_1/axis:output:0*
T0*1
_output_shapes
:€€€€€€€€€ОО є
1sequential/rot_equiv_conv2d/Rank_4/ReadVariableOpReadVariableOp?sequential_rot_equiv_conv2d_convolution_readvariableop_resource*&
_output_shapes
: *
dtype0d
"sequential/rot_equiv_conv2d/Rank_4Const*
_output_shapes
: *
dtype0*
value	B :і
,sequential/rot_equiv_conv2d/ReadVariableOp_1ReadVariableOp?sequential_rot_equiv_conv2d_convolution_readvariableop_resource*&
_output_shapes
: *
dtype0d
"sequential/rot_equiv_conv2d/Rank_5Const*
_output_shapes
: *
dtype0*
value	B :v
,sequential/rot_equiv_conv2d/ReverseV2_2/axisConst*
_output_shapes
:*
dtype0*
valueB: “
'sequential/rot_equiv_conv2d/ReverseV2_2	ReverseV24sequential/rot_equiv_conv2d/ReadVariableOp_1:value:05sequential/rot_equiv_conv2d/ReverseV2_2/axis:output:0*
T0*&
_output_shapes
: d
"sequential/rot_equiv_conv2d/Rank_6Const*
_output_shapes
: *
dtype0*
value	B :v
,sequential/rot_equiv_conv2d/ReverseV2_3/axisConst*
_output_shapes
:*
dtype0*
valueB:ќ
'sequential/rot_equiv_conv2d/ReverseV2_3	ReverseV20sequential/rot_equiv_conv2d/ReverseV2_2:output:05sequential/rot_equiv_conv2d/ReverseV2_3/axis:output:0*
T0*&
_output_shapes
: в
)sequential/rot_equiv_conv2d/convolution_2Conv2Drot_equiv_conv2d_input0sequential/rot_equiv_conv2d/ReverseV2_3:output:0*
T0*1
_output_shapes
:€€€€€€€€€ОО *
paddingVALID*
strides
d
"sequential/rot_equiv_conv2d/Rank_7Const*
_output_shapes
: *
dtype0*
value	B :d
"sequential/rot_equiv_conv2d/Rank_8Const*
_output_shapes
: *
dtype0*
value	B :v
,sequential/rot_equiv_conv2d/ReverseV2_4/axisConst*
_output_shapes
:*
dtype0*
valueB:џ
'sequential/rot_equiv_conv2d/ReverseV2_4	ReverseV22sequential/rot_equiv_conv2d/convolution_2:output:05sequential/rot_equiv_conv2d/ReverseV2_4/axis:output:0*
T0*1
_output_shapes
:€€€€€€€€€ОО d
"sequential/rot_equiv_conv2d/Rank_9Const*
_output_shapes
: *
dtype0*
value	B :v
,sequential/rot_equiv_conv2d/ReverseV2_5/axisConst*
_output_shapes
:*
dtype0*
valueB:ў
'sequential/rot_equiv_conv2d/ReverseV2_5	ReverseV20sequential/rot_equiv_conv2d/ReverseV2_4:output:05sequential/rot_equiv_conv2d/ReverseV2_5/axis:output:0*
T0*1
_output_shapes
:€€€€€€€€€ОО Ї
2sequential/rot_equiv_conv2d/Rank_10/ReadVariableOpReadVariableOp?sequential_rot_equiv_conv2d_convolution_readvariableop_resource*&
_output_shapes
: *
dtype0e
#sequential/rot_equiv_conv2d/Rank_10Const*
_output_shapes
: *
dtype0*
value	B :k
)sequential/rot_equiv_conv2d/range_2/startConst*
_output_shapes
: *
dtype0*
value	B : k
)sequential/rot_equiv_conv2d/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :ё
#sequential/rot_equiv_conv2d/range_2Range2sequential/rot_equiv_conv2d/range_2/start:output:0,sequential/rot_equiv_conv2d/Rank_10:output:02sequential/rot_equiv_conv2d/range_2/delta:output:0*
_output_shapes
:Т
9sequential/rot_equiv_conv2d/TensorScatterUpdate_2/indicesConst*
_output_shapes

:*
dtype0*!
valueB"       К
9sequential/rot_equiv_conv2d/TensorScatterUpdate_2/updatesConst*
_output_shapes
:*
dtype0*
valueB"       ≥
1sequential/rot_equiv_conv2d/TensorScatterUpdate_2TensorScatterUpdate,sequential/rot_equiv_conv2d/range_2:output:0Bsequential/rot_equiv_conv2d/TensorScatterUpdate_2/indices:output:0Bsequential/rot_equiv_conv2d/TensorScatterUpdate_2/updates:output:0*
T0*
Tindices0*
_output_shapes
:і
,sequential/rot_equiv_conv2d/ReadVariableOp_2ReadVariableOp?sequential_rot_equiv_conv2d_convolution_readvariableop_resource*&
_output_shapes
: *
dtype0„
'sequential/rot_equiv_conv2d/transpose_2	Transpose4sequential/rot_equiv_conv2d/ReadVariableOp_2:value:0:sequential/rot_equiv_conv2d/TensorScatterUpdate_2:output:0*
T0*&
_output_shapes
: e
#sequential/rot_equiv_conv2d/Rank_11Const*
_output_shapes
: *
dtype0*
value	B :v
,sequential/rot_equiv_conv2d/ReverseV2_6/axisConst*
_output_shapes
:*
dtype0*
valueB:…
'sequential/rot_equiv_conv2d/ReverseV2_6	ReverseV2+sequential/rot_equiv_conv2d/transpose_2:y:05sequential/rot_equiv_conv2d/ReverseV2_6/axis:output:0*
T0*&
_output_shapes
: в
)sequential/rot_equiv_conv2d/convolution_3Conv2Drot_equiv_conv2d_input0sequential/rot_equiv_conv2d/ReverseV2_6:output:0*
T0*1
_output_shapes
:€€€€€€€€€ОО *
paddingVALID*
strides
e
#sequential/rot_equiv_conv2d/Rank_12Const*
_output_shapes
: *
dtype0*
value	B :k
)sequential/rot_equiv_conv2d/range_3/startConst*
_output_shapes
: *
dtype0*
value	B : k
)sequential/rot_equiv_conv2d/range_3/deltaConst*
_output_shapes
: *
dtype0*
value	B :ё
#sequential/rot_equiv_conv2d/range_3Range2sequential/rot_equiv_conv2d/range_3/start:output:0,sequential/rot_equiv_conv2d/Rank_12:output:02sequential/rot_equiv_conv2d/range_3/delta:output:0*
_output_shapes
:Т
9sequential/rot_equiv_conv2d/TensorScatterUpdate_3/indicesConst*
_output_shapes

:*
dtype0*!
valueB"      К
9sequential/rot_equiv_conv2d/TensorScatterUpdate_3/updatesConst*
_output_shapes
:*
dtype0*
valueB"      ≥
1sequential/rot_equiv_conv2d/TensorScatterUpdate_3TensorScatterUpdate,sequential/rot_equiv_conv2d/range_3:output:0Bsequential/rot_equiv_conv2d/TensorScatterUpdate_3/indices:output:0Bsequential/rot_equiv_conv2d/TensorScatterUpdate_3/updates:output:0*
T0*
Tindices0*
_output_shapes
:e
#sequential/rot_equiv_conv2d/Rank_13Const*
_output_shapes
: *
dtype0*
value	B :v
,sequential/rot_equiv_conv2d/ReverseV2_7/axisConst*
_output_shapes
:*
dtype0*
valueB:џ
'sequential/rot_equiv_conv2d/ReverseV2_7	ReverseV22sequential/rot_equiv_conv2d/convolution_3:output:05sequential/rot_equiv_conv2d/ReverseV2_7/axis:output:0*
T0*1
_output_shapes
:€€€€€€€€€ОО ё
'sequential/rot_equiv_conv2d/transpose_3	Transpose0sequential/rot_equiv_conv2d/ReverseV2_7:output:0:sequential/rot_equiv_conv2d/TensorScatterUpdate_3:output:0*
T0*1
_output_shapes
:€€€€€€€€€ОО  
!sequential/rot_equiv_conv2d/stackPack0sequential/rot_equiv_conv2d/convolution:output:00sequential/rot_equiv_conv2d/ReverseV2_1:output:00sequential/rot_equiv_conv2d/ReverseV2_5:output:0+sequential/rot_equiv_conv2d/transpose_3:y:0*
N*
T0*5
_output_shapes#
!:€€€€€€€€€ОО *
axisю€€€€€€€€Ф
 sequential/rot_equiv_conv2d/ReluRelu*sequential/rot_equiv_conv2d/stack:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€ОО ™
2sequential/rot_equiv_conv2d/BiasAdd/ReadVariableOpReadVariableOp;sequential_rot_equiv_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Џ
#sequential/rot_equiv_conv2d/BiasAddBiasAdd.sequential/rot_equiv_conv2d/Relu:activations:0:sequential/rot_equiv_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:€€€€€€€€€ОО c
!sequential/rot_equiv_pool2d/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R Н
!sequential/rot_equiv_pool2d/ShapeShape,sequential/rot_equiv_conv2d/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	В
/sequential/rot_equiv_pool2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€Д
1sequential/rot_equiv_pool2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€{
1sequential/rot_equiv_pool2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ё
)sequential/rot_equiv_pool2d/strided_sliceStridedSlice*sequential/rot_equiv_pool2d/Shape:output:08sequential/rot_equiv_pool2d/strided_slice/stack:output:0:sequential/rot_equiv_pool2d/strided_slice/stack_1:output:0:sequential/rot_equiv_pool2d/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskc
!sequential/rot_equiv_pool2d/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RІ
sequential/rot_equiv_pool2d/subSub2sequential/rot_equiv_pool2d/strided_slice:output:0*sequential/rot_equiv_pool2d/sub/y:output:0*
T0	*
_output_shapes
: Ѓ
1sequential/rot_equiv_pool2d/clip_by_value/MinimumMinimum*sequential/rot_equiv_pool2d/Const:output:0#sequential/rot_equiv_pool2d/sub:z:0*
T0	*
_output_shapes
: m
+sequential/rot_equiv_pool2d/clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ¬
)sequential/rot_equiv_pool2d/clip_by_valueMaximum5sequential/rot_equiv_pool2d/clip_by_value/Minimum:z:04sequential/rot_equiv_pool2d/clip_by_value/y:output:0*
T0	*
_output_shapes
: t
)sequential/rot_equiv_pool2d/GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€†
$sequential/rot_equiv_pool2d/GatherV2GatherV2,sequential/rot_equiv_conv2d/BiasAdd:output:0-sequential/rot_equiv_pool2d/clip_by_value:z:02sequential/rot_equiv_pool2d/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*1
_output_shapes
:€€€€€€€€€ОО Ў
1sequential/rot_equiv_pool2d/max_pooling2d/MaxPoolMaxPool-sequential/rot_equiv_pool2d/GatherV2:output:0*/
_output_shapes
:€€€€€€€€€GG *
ksize
*
paddingVALID*
strides
e
#sequential/rot_equiv_pool2d/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 RП
#sequential/rot_equiv_pool2d/Shape_1Shape,sequential/rot_equiv_conv2d/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	Д
1sequential/rot_equiv_pool2d/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€Ж
3sequential/rot_equiv_pool2d/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€}
3sequential/rot_equiv_pool2d/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:з
+sequential/rot_equiv_pool2d/strided_slice_1StridedSlice,sequential/rot_equiv_pool2d/Shape_1:output:0:sequential/rot_equiv_pool2d/strided_slice_1/stack:output:0<sequential/rot_equiv_pool2d/strided_slice_1/stack_1:output:0<sequential/rot_equiv_pool2d/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maske
#sequential/rot_equiv_pool2d/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R≠
!sequential/rot_equiv_pool2d/sub_1Sub4sequential/rot_equiv_pool2d/strided_slice_1:output:0,sequential/rot_equiv_pool2d/sub_1/y:output:0*
T0	*
_output_shapes
: і
3sequential/rot_equiv_pool2d/clip_by_value_1/MinimumMinimum,sequential/rot_equiv_pool2d/Const_1:output:0%sequential/rot_equiv_pool2d/sub_1:z:0*
T0	*
_output_shapes
: o
-sequential/rot_equiv_pool2d/clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R »
+sequential/rot_equiv_pool2d/clip_by_value_1Maximum7sequential/rot_equiv_pool2d/clip_by_value_1/Minimum:z:06sequential/rot_equiv_pool2d/clip_by_value_1/y:output:0*
T0	*
_output_shapes
: v
+sequential/rot_equiv_pool2d/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€¶
&sequential/rot_equiv_pool2d/GatherV2_1GatherV2,sequential/rot_equiv_conv2d/BiasAdd:output:0/sequential/rot_equiv_pool2d/clip_by_value_1:z:04sequential/rot_equiv_pool2d/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*1
_output_shapes
:€€€€€€€€€ОО №
3sequential/rot_equiv_pool2d/max_pooling2d/MaxPool_1MaxPool/sequential/rot_equiv_pool2d/GatherV2_1:output:0*/
_output_shapes
:€€€€€€€€€GG *
ksize
*
paddingVALID*
strides
e
#sequential/rot_equiv_pool2d/Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 RП
#sequential/rot_equiv_pool2d/Shape_2Shape,sequential/rot_equiv_conv2d/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	Д
1sequential/rot_equiv_pool2d/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€Ж
3sequential/rot_equiv_pool2d/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€}
3sequential/rot_equiv_pool2d/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:з
+sequential/rot_equiv_pool2d/strided_slice_2StridedSlice,sequential/rot_equiv_pool2d/Shape_2:output:0:sequential/rot_equiv_pool2d/strided_slice_2/stack:output:0<sequential/rot_equiv_pool2d/strided_slice_2/stack_1:output:0<sequential/rot_equiv_pool2d/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maske
#sequential/rot_equiv_pool2d/sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R≠
!sequential/rot_equiv_pool2d/sub_2Sub4sequential/rot_equiv_pool2d/strided_slice_2:output:0,sequential/rot_equiv_pool2d/sub_2/y:output:0*
T0	*
_output_shapes
: і
3sequential/rot_equiv_pool2d/clip_by_value_2/MinimumMinimum,sequential/rot_equiv_pool2d/Const_2:output:0%sequential/rot_equiv_pool2d/sub_2:z:0*
T0	*
_output_shapes
: o
-sequential/rot_equiv_pool2d/clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R »
+sequential/rot_equiv_pool2d/clip_by_value_2Maximum7sequential/rot_equiv_pool2d/clip_by_value_2/Minimum:z:06sequential/rot_equiv_pool2d/clip_by_value_2/y:output:0*
T0	*
_output_shapes
: v
+sequential/rot_equiv_pool2d/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€¶
&sequential/rot_equiv_pool2d/GatherV2_2GatherV2,sequential/rot_equiv_conv2d/BiasAdd:output:0/sequential/rot_equiv_pool2d/clip_by_value_2:z:04sequential/rot_equiv_pool2d/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*1
_output_shapes
:€€€€€€€€€ОО №
3sequential/rot_equiv_pool2d/max_pooling2d/MaxPool_2MaxPool/sequential/rot_equiv_pool2d/GatherV2_2:output:0*/
_output_shapes
:€€€€€€€€€GG *
ksize
*
paddingVALID*
strides
e
#sequential/rot_equiv_pool2d/Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 RП
#sequential/rot_equiv_pool2d/Shape_3Shape,sequential/rot_equiv_conv2d/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	Д
1sequential/rot_equiv_pool2d/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€Ж
3sequential/rot_equiv_pool2d/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€}
3sequential/rot_equiv_pool2d/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:з
+sequential/rot_equiv_pool2d/strided_slice_3StridedSlice,sequential/rot_equiv_pool2d/Shape_3:output:0:sequential/rot_equiv_pool2d/strided_slice_3/stack:output:0<sequential/rot_equiv_pool2d/strided_slice_3/stack_1:output:0<sequential/rot_equiv_pool2d/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maske
#sequential/rot_equiv_pool2d/sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R≠
!sequential/rot_equiv_pool2d/sub_3Sub4sequential/rot_equiv_pool2d/strided_slice_3:output:0,sequential/rot_equiv_pool2d/sub_3/y:output:0*
T0	*
_output_shapes
: і
3sequential/rot_equiv_pool2d/clip_by_value_3/MinimumMinimum,sequential/rot_equiv_pool2d/Const_3:output:0%sequential/rot_equiv_pool2d/sub_3:z:0*
T0	*
_output_shapes
: o
-sequential/rot_equiv_pool2d/clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R »
+sequential/rot_equiv_pool2d/clip_by_value_3Maximum7sequential/rot_equiv_pool2d/clip_by_value_3/Minimum:z:06sequential/rot_equiv_pool2d/clip_by_value_3/y:output:0*
T0	*
_output_shapes
: v
+sequential/rot_equiv_pool2d/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€¶
&sequential/rot_equiv_pool2d/GatherV2_3GatherV2,sequential/rot_equiv_conv2d/BiasAdd:output:0/sequential/rot_equiv_pool2d/clip_by_value_3:z:04sequential/rot_equiv_pool2d/GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*1
_output_shapes
:€€€€€€€€€ОО №
3sequential/rot_equiv_pool2d/max_pooling2d/MaxPool_3MaxPool/sequential/rot_equiv_pool2d/GatherV2_3:output:0*/
_output_shapes
:€€€€€€€€€GG *
ksize
*
paddingVALID*
strides
ы
!sequential/rot_equiv_pool2d/stackPack:sequential/rot_equiv_pool2d/max_pooling2d/MaxPool:output:0<sequential/rot_equiv_pool2d/max_pooling2d/MaxPool_1:output:0<sequential/rot_equiv_pool2d/max_pooling2d/MaxPool_2:output:0<sequential/rot_equiv_pool2d/max_pooling2d/MaxPool_3:output:0*
N*
T0*3
_output_shapes!
:€€€€€€€€€GG *
axisю€€€€€€€€e
#sequential/rot_equiv_conv2d_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R Н
#sequential/rot_equiv_conv2d_1/ShapeShape*sequential/rot_equiv_pool2d/stack:output:0*
T0*
_output_shapes
:*
out_type0	Д
1sequential/rot_equiv_conv2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€Ж
3sequential/rot_equiv_conv2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€}
3sequential/rot_equiv_conv2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:з
+sequential/rot_equiv_conv2d_1/strided_sliceStridedSlice,sequential/rot_equiv_conv2d_1/Shape:output:0:sequential/rot_equiv_conv2d_1/strided_slice/stack:output:0<sequential/rot_equiv_conv2d_1/strided_slice/stack_1:output:0<sequential/rot_equiv_conv2d_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maske
#sequential/rot_equiv_conv2d_1/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R≠
!sequential/rot_equiv_conv2d_1/subSub4sequential/rot_equiv_conv2d_1/strided_slice:output:0,sequential/rot_equiv_conv2d_1/sub/y:output:0*
T0	*
_output_shapes
: і
3sequential/rot_equiv_conv2d_1/clip_by_value/MinimumMinimum,sequential/rot_equiv_conv2d_1/Const:output:0%sequential/rot_equiv_conv2d_1/sub:z:0*
T0	*
_output_shapes
: o
-sequential/rot_equiv_conv2d_1/clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R »
+sequential/rot_equiv_conv2d_1/clip_by_valueMaximum7sequential/rot_equiv_conv2d_1/clip_by_value/Minimum:z:06sequential/rot_equiv_conv2d_1/clip_by_value/y:output:0*
T0	*
_output_shapes
: v
+sequential/rot_equiv_conv2d_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€Ґ
&sequential/rot_equiv_conv2d_1/GatherV2GatherV2*sequential/rot_equiv_pool2d/stack:output:0/sequential/rot_equiv_conv2d_1/clip_by_value:z:04sequential/rot_equiv_conv2d_1/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€GG ¬
8sequential/rot_equiv_conv2d_1/convolution/ReadVariableOpReadVariableOpAsequential_rot_equiv_conv2d_1_convolution_readvariableop_resource*&
_output_shapes
:  *
dtype0Й
)sequential/rot_equiv_conv2d_1/convolutionConv2D/sequential/rot_equiv_conv2d_1/GatherV2:output:0@sequential/rot_equiv_conv2d_1/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€EE *
paddingVALID*
strides
g
%sequential/rot_equiv_conv2d_1/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 RП
%sequential/rot_equiv_conv2d_1/Shape_1Shape*sequential/rot_equiv_pool2d/stack:output:0*
T0*
_output_shapes
:*
out_type0	Ж
3sequential/rot_equiv_conv2d_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€И
5sequential/rot_equiv_conv2d_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€
5sequential/rot_equiv_conv2d_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:с
-sequential/rot_equiv_conv2d_1/strided_slice_1StridedSlice.sequential/rot_equiv_conv2d_1/Shape_1:output:0<sequential/rot_equiv_conv2d_1/strided_slice_1/stack:output:0>sequential/rot_equiv_conv2d_1/strided_slice_1/stack_1:output:0>sequential/rot_equiv_conv2d_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskg
%sequential/rot_equiv_conv2d_1/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R≥
#sequential/rot_equiv_conv2d_1/sub_1Sub6sequential/rot_equiv_conv2d_1/strided_slice_1:output:0.sequential/rot_equiv_conv2d_1/sub_1/y:output:0*
T0	*
_output_shapes
: Ї
5sequential/rot_equiv_conv2d_1/clip_by_value_1/MinimumMinimum.sequential/rot_equiv_conv2d_1/Const_1:output:0'sequential/rot_equiv_conv2d_1/sub_1:z:0*
T0	*
_output_shapes
: q
/sequential/rot_equiv_conv2d_1/clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ќ
-sequential/rot_equiv_conv2d_1/clip_by_value_1Maximum9sequential/rot_equiv_conv2d_1/clip_by_value_1/Minimum:z:08sequential/rot_equiv_conv2d_1/clip_by_value_1/y:output:0*
T0	*
_output_shapes
: x
-sequential/rot_equiv_conv2d_1/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€®
(sequential/rot_equiv_conv2d_1/GatherV2_1GatherV2*sequential/rot_equiv_pool2d/stack:output:01sequential/rot_equiv_conv2d_1/clip_by_value_1:z:06sequential/rot_equiv_conv2d_1/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€GG ƒ
:sequential/rot_equiv_conv2d_1/convolution_1/ReadVariableOpReadVariableOpAsequential_rot_equiv_conv2d_1_convolution_readvariableop_resource*&
_output_shapes
:  *
dtype0П
+sequential/rot_equiv_conv2d_1/convolution_1Conv2D1sequential/rot_equiv_conv2d_1/GatherV2_1:output:0Bsequential/rot_equiv_conv2d_1/convolution_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€EE *
paddingVALID*
strides
g
%sequential/rot_equiv_conv2d_1/Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 RП
%sequential/rot_equiv_conv2d_1/Shape_2Shape*sequential/rot_equiv_pool2d/stack:output:0*
T0*
_output_shapes
:*
out_type0	Ж
3sequential/rot_equiv_conv2d_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€И
5sequential/rot_equiv_conv2d_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€
5sequential/rot_equiv_conv2d_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:с
-sequential/rot_equiv_conv2d_1/strided_slice_2StridedSlice.sequential/rot_equiv_conv2d_1/Shape_2:output:0<sequential/rot_equiv_conv2d_1/strided_slice_2/stack:output:0>sequential/rot_equiv_conv2d_1/strided_slice_2/stack_1:output:0>sequential/rot_equiv_conv2d_1/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskg
%sequential/rot_equiv_conv2d_1/sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R≥
#sequential/rot_equiv_conv2d_1/sub_2Sub6sequential/rot_equiv_conv2d_1/strided_slice_2:output:0.sequential/rot_equiv_conv2d_1/sub_2/y:output:0*
T0	*
_output_shapes
: Ї
5sequential/rot_equiv_conv2d_1/clip_by_value_2/MinimumMinimum.sequential/rot_equiv_conv2d_1/Const_2:output:0'sequential/rot_equiv_conv2d_1/sub_2:z:0*
T0	*
_output_shapes
: q
/sequential/rot_equiv_conv2d_1/clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ќ
-sequential/rot_equiv_conv2d_1/clip_by_value_2Maximum9sequential/rot_equiv_conv2d_1/clip_by_value_2/Minimum:z:08sequential/rot_equiv_conv2d_1/clip_by_value_2/y:output:0*
T0	*
_output_shapes
: x
-sequential/rot_equiv_conv2d_1/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€®
(sequential/rot_equiv_conv2d_1/GatherV2_2GatherV2*sequential/rot_equiv_pool2d/stack:output:01sequential/rot_equiv_conv2d_1/clip_by_value_2:z:06sequential/rot_equiv_conv2d_1/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€GG ƒ
:sequential/rot_equiv_conv2d_1/convolution_2/ReadVariableOpReadVariableOpAsequential_rot_equiv_conv2d_1_convolution_readvariableop_resource*&
_output_shapes
:  *
dtype0П
+sequential/rot_equiv_conv2d_1/convolution_2Conv2D1sequential/rot_equiv_conv2d_1/GatherV2_2:output:0Bsequential/rot_equiv_conv2d_1/convolution_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€EE *
paddingVALID*
strides
g
%sequential/rot_equiv_conv2d_1/Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 RП
%sequential/rot_equiv_conv2d_1/Shape_3Shape*sequential/rot_equiv_pool2d/stack:output:0*
T0*
_output_shapes
:*
out_type0	Ж
3sequential/rot_equiv_conv2d_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€И
5sequential/rot_equiv_conv2d_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€
5sequential/rot_equiv_conv2d_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:с
-sequential/rot_equiv_conv2d_1/strided_slice_3StridedSlice.sequential/rot_equiv_conv2d_1/Shape_3:output:0<sequential/rot_equiv_conv2d_1/strided_slice_3/stack:output:0>sequential/rot_equiv_conv2d_1/strided_slice_3/stack_1:output:0>sequential/rot_equiv_conv2d_1/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskg
%sequential/rot_equiv_conv2d_1/sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R≥
#sequential/rot_equiv_conv2d_1/sub_3Sub6sequential/rot_equiv_conv2d_1/strided_slice_3:output:0.sequential/rot_equiv_conv2d_1/sub_3/y:output:0*
T0	*
_output_shapes
: Ї
5sequential/rot_equiv_conv2d_1/clip_by_value_3/MinimumMinimum.sequential/rot_equiv_conv2d_1/Const_3:output:0'sequential/rot_equiv_conv2d_1/sub_3:z:0*
T0	*
_output_shapes
: q
/sequential/rot_equiv_conv2d_1/clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ќ
-sequential/rot_equiv_conv2d_1/clip_by_value_3Maximum9sequential/rot_equiv_conv2d_1/clip_by_value_3/Minimum:z:08sequential/rot_equiv_conv2d_1/clip_by_value_3/y:output:0*
T0	*
_output_shapes
: x
-sequential/rot_equiv_conv2d_1/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€®
(sequential/rot_equiv_conv2d_1/GatherV2_3GatherV2*sequential/rot_equiv_pool2d/stack:output:01sequential/rot_equiv_conv2d_1/clip_by_value_3:z:06sequential/rot_equiv_conv2d_1/GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€GG ƒ
:sequential/rot_equiv_conv2d_1/convolution_3/ReadVariableOpReadVariableOpAsequential_rot_equiv_conv2d_1_convolution_readvariableop_resource*&
_output_shapes
:  *
dtype0П
+sequential/rot_equiv_conv2d_1/convolution_3Conv2D1sequential/rot_equiv_conv2d_1/GatherV2_3:output:0Bsequential/rot_equiv_conv2d_1/convolution_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€EE *
paddingVALID*
strides
Ё
#sequential/rot_equiv_conv2d_1/stackPack2sequential/rot_equiv_conv2d_1/convolution:output:04sequential/rot_equiv_conv2d_1/convolution_1:output:04sequential/rot_equiv_conv2d_1/convolution_2:output:04sequential/rot_equiv_conv2d_1/convolution_3:output:0*
N*
T0*3
_output_shapes!
:€€€€€€€€€EE *
axisю€€€€€€€€Ц
"sequential/rot_equiv_conv2d_1/ReluRelu,sequential/rot_equiv_conv2d_1/stack:output:0*
T0*3
_output_shapes!
:€€€€€€€€€EE Ѓ
4sequential/rot_equiv_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp=sequential_rot_equiv_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ё
%sequential/rot_equiv_conv2d_1/BiasAddBiasAdd0sequential/rot_equiv_conv2d_1/Relu:activations:0<sequential/rot_equiv_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€EE e
#sequential/rot_equiv_pool2d_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R С
#sequential/rot_equiv_pool2d_1/ShapeShape.sequential/rot_equiv_conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	Д
1sequential/rot_equiv_pool2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€Ж
3sequential/rot_equiv_pool2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€}
3sequential/rot_equiv_pool2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:з
+sequential/rot_equiv_pool2d_1/strided_sliceStridedSlice,sequential/rot_equiv_pool2d_1/Shape:output:0:sequential/rot_equiv_pool2d_1/strided_slice/stack:output:0<sequential/rot_equiv_pool2d_1/strided_slice/stack_1:output:0<sequential/rot_equiv_pool2d_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maske
#sequential/rot_equiv_pool2d_1/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R≠
!sequential/rot_equiv_pool2d_1/subSub4sequential/rot_equiv_pool2d_1/strided_slice:output:0,sequential/rot_equiv_pool2d_1/sub/y:output:0*
T0	*
_output_shapes
: і
3sequential/rot_equiv_pool2d_1/clip_by_value/MinimumMinimum,sequential/rot_equiv_pool2d_1/Const:output:0%sequential/rot_equiv_pool2d_1/sub:z:0*
T0	*
_output_shapes
: o
-sequential/rot_equiv_pool2d_1/clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R »
+sequential/rot_equiv_pool2d_1/clip_by_valueMaximum7sequential/rot_equiv_pool2d_1/clip_by_value/Minimum:z:06sequential/rot_equiv_pool2d_1/clip_by_value/y:output:0*
T0	*
_output_shapes
: v
+sequential/rot_equiv_pool2d_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€¶
&sequential/rot_equiv_pool2d_1/GatherV2GatherV2.sequential/rot_equiv_conv2d_1/BiasAdd:output:0/sequential/rot_equiv_pool2d_1/clip_by_value:z:04sequential/rot_equiv_pool2d_1/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€EE ё
5sequential/rot_equiv_pool2d_1/max_pooling2d_1/MaxPoolMaxPool/sequential/rot_equiv_pool2d_1/GatherV2:output:0*/
_output_shapes
:€€€€€€€€€"" *
ksize
*
paddingVALID*
strides
g
%sequential/rot_equiv_pool2d_1/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 RУ
%sequential/rot_equiv_pool2d_1/Shape_1Shape.sequential/rot_equiv_conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	Ж
3sequential/rot_equiv_pool2d_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€И
5sequential/rot_equiv_pool2d_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€
5sequential/rot_equiv_pool2d_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:с
-sequential/rot_equiv_pool2d_1/strided_slice_1StridedSlice.sequential/rot_equiv_pool2d_1/Shape_1:output:0<sequential/rot_equiv_pool2d_1/strided_slice_1/stack:output:0>sequential/rot_equiv_pool2d_1/strided_slice_1/stack_1:output:0>sequential/rot_equiv_pool2d_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskg
%sequential/rot_equiv_pool2d_1/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R≥
#sequential/rot_equiv_pool2d_1/sub_1Sub6sequential/rot_equiv_pool2d_1/strided_slice_1:output:0.sequential/rot_equiv_pool2d_1/sub_1/y:output:0*
T0	*
_output_shapes
: Ї
5sequential/rot_equiv_pool2d_1/clip_by_value_1/MinimumMinimum.sequential/rot_equiv_pool2d_1/Const_1:output:0'sequential/rot_equiv_pool2d_1/sub_1:z:0*
T0	*
_output_shapes
: q
/sequential/rot_equiv_pool2d_1/clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ќ
-sequential/rot_equiv_pool2d_1/clip_by_value_1Maximum9sequential/rot_equiv_pool2d_1/clip_by_value_1/Minimum:z:08sequential/rot_equiv_pool2d_1/clip_by_value_1/y:output:0*
T0	*
_output_shapes
: x
-sequential/rot_equiv_pool2d_1/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ђ
(sequential/rot_equiv_pool2d_1/GatherV2_1GatherV2.sequential/rot_equiv_conv2d_1/BiasAdd:output:01sequential/rot_equiv_pool2d_1/clip_by_value_1:z:06sequential/rot_equiv_pool2d_1/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€EE в
7sequential/rot_equiv_pool2d_1/max_pooling2d_1/MaxPool_1MaxPool1sequential/rot_equiv_pool2d_1/GatherV2_1:output:0*/
_output_shapes
:€€€€€€€€€"" *
ksize
*
paddingVALID*
strides
g
%sequential/rot_equiv_pool2d_1/Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 RУ
%sequential/rot_equiv_pool2d_1/Shape_2Shape.sequential/rot_equiv_conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	Ж
3sequential/rot_equiv_pool2d_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€И
5sequential/rot_equiv_pool2d_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€
5sequential/rot_equiv_pool2d_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:с
-sequential/rot_equiv_pool2d_1/strided_slice_2StridedSlice.sequential/rot_equiv_pool2d_1/Shape_2:output:0<sequential/rot_equiv_pool2d_1/strided_slice_2/stack:output:0>sequential/rot_equiv_pool2d_1/strided_slice_2/stack_1:output:0>sequential/rot_equiv_pool2d_1/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskg
%sequential/rot_equiv_pool2d_1/sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R≥
#sequential/rot_equiv_pool2d_1/sub_2Sub6sequential/rot_equiv_pool2d_1/strided_slice_2:output:0.sequential/rot_equiv_pool2d_1/sub_2/y:output:0*
T0	*
_output_shapes
: Ї
5sequential/rot_equiv_pool2d_1/clip_by_value_2/MinimumMinimum.sequential/rot_equiv_pool2d_1/Const_2:output:0'sequential/rot_equiv_pool2d_1/sub_2:z:0*
T0	*
_output_shapes
: q
/sequential/rot_equiv_pool2d_1/clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ќ
-sequential/rot_equiv_pool2d_1/clip_by_value_2Maximum9sequential/rot_equiv_pool2d_1/clip_by_value_2/Minimum:z:08sequential/rot_equiv_pool2d_1/clip_by_value_2/y:output:0*
T0	*
_output_shapes
: x
-sequential/rot_equiv_pool2d_1/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ђ
(sequential/rot_equiv_pool2d_1/GatherV2_2GatherV2.sequential/rot_equiv_conv2d_1/BiasAdd:output:01sequential/rot_equiv_pool2d_1/clip_by_value_2:z:06sequential/rot_equiv_pool2d_1/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€EE в
7sequential/rot_equiv_pool2d_1/max_pooling2d_1/MaxPool_2MaxPool1sequential/rot_equiv_pool2d_1/GatherV2_2:output:0*/
_output_shapes
:€€€€€€€€€"" *
ksize
*
paddingVALID*
strides
g
%sequential/rot_equiv_pool2d_1/Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 RУ
%sequential/rot_equiv_pool2d_1/Shape_3Shape.sequential/rot_equiv_conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	Ж
3sequential/rot_equiv_pool2d_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€И
5sequential/rot_equiv_pool2d_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€
5sequential/rot_equiv_pool2d_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:с
-sequential/rot_equiv_pool2d_1/strided_slice_3StridedSlice.sequential/rot_equiv_pool2d_1/Shape_3:output:0<sequential/rot_equiv_pool2d_1/strided_slice_3/stack:output:0>sequential/rot_equiv_pool2d_1/strided_slice_3/stack_1:output:0>sequential/rot_equiv_pool2d_1/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskg
%sequential/rot_equiv_pool2d_1/sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R≥
#sequential/rot_equiv_pool2d_1/sub_3Sub6sequential/rot_equiv_pool2d_1/strided_slice_3:output:0.sequential/rot_equiv_pool2d_1/sub_3/y:output:0*
T0	*
_output_shapes
: Ї
5sequential/rot_equiv_pool2d_1/clip_by_value_3/MinimumMinimum.sequential/rot_equiv_pool2d_1/Const_3:output:0'sequential/rot_equiv_pool2d_1/sub_3:z:0*
T0	*
_output_shapes
: q
/sequential/rot_equiv_pool2d_1/clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ќ
-sequential/rot_equiv_pool2d_1/clip_by_value_3Maximum9sequential/rot_equiv_pool2d_1/clip_by_value_3/Minimum:z:08sequential/rot_equiv_pool2d_1/clip_by_value_3/y:output:0*
T0	*
_output_shapes
: x
-sequential/rot_equiv_pool2d_1/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ђ
(sequential/rot_equiv_pool2d_1/GatherV2_3GatherV2.sequential/rot_equiv_conv2d_1/BiasAdd:output:01sequential/rot_equiv_pool2d_1/clip_by_value_3:z:06sequential/rot_equiv_pool2d_1/GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€EE в
7sequential/rot_equiv_pool2d_1/max_pooling2d_1/MaxPool_3MaxPool1sequential/rot_equiv_pool2d_1/GatherV2_3:output:0*/
_output_shapes
:€€€€€€€€€"" *
ksize
*
paddingVALID*
strides
Н
#sequential/rot_equiv_pool2d_1/stackPack>sequential/rot_equiv_pool2d_1/max_pooling2d_1/MaxPool:output:0@sequential/rot_equiv_pool2d_1/max_pooling2d_1/MaxPool_1:output:0@sequential/rot_equiv_pool2d_1/max_pooling2d_1/MaxPool_2:output:0@sequential/rot_equiv_pool2d_1/max_pooling2d_1/MaxPool_3:output:0*
N*
T0*3
_output_shapes!
:€€€€€€€€€"" *
axisю€€€€€€€€e
#sequential/rot_equiv_conv2d_2/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R П
#sequential/rot_equiv_conv2d_2/ShapeShape,sequential/rot_equiv_pool2d_1/stack:output:0*
T0*
_output_shapes
:*
out_type0	Д
1sequential/rot_equiv_conv2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€Ж
3sequential/rot_equiv_conv2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€}
3sequential/rot_equiv_conv2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:з
+sequential/rot_equiv_conv2d_2/strided_sliceStridedSlice,sequential/rot_equiv_conv2d_2/Shape:output:0:sequential/rot_equiv_conv2d_2/strided_slice/stack:output:0<sequential/rot_equiv_conv2d_2/strided_slice/stack_1:output:0<sequential/rot_equiv_conv2d_2/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maske
#sequential/rot_equiv_conv2d_2/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R≠
!sequential/rot_equiv_conv2d_2/subSub4sequential/rot_equiv_conv2d_2/strided_slice:output:0,sequential/rot_equiv_conv2d_2/sub/y:output:0*
T0	*
_output_shapes
: і
3sequential/rot_equiv_conv2d_2/clip_by_value/MinimumMinimum,sequential/rot_equiv_conv2d_2/Const:output:0%sequential/rot_equiv_conv2d_2/sub:z:0*
T0	*
_output_shapes
: o
-sequential/rot_equiv_conv2d_2/clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R »
+sequential/rot_equiv_conv2d_2/clip_by_valueMaximum7sequential/rot_equiv_conv2d_2/clip_by_value/Minimum:z:06sequential/rot_equiv_conv2d_2/clip_by_value/y:output:0*
T0	*
_output_shapes
: v
+sequential/rot_equiv_conv2d_2/GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€§
&sequential/rot_equiv_conv2d_2/GatherV2GatherV2,sequential/rot_equiv_pool2d_1/stack:output:0/sequential/rot_equiv_conv2d_2/clip_by_value:z:04sequential/rot_equiv_conv2d_2/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€"" ¬
8sequential/rot_equiv_conv2d_2/convolution/ReadVariableOpReadVariableOpAsequential_rot_equiv_conv2d_2_convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0Й
)sequential/rot_equiv_conv2d_2/convolutionConv2D/sequential/rot_equiv_conv2d_2/GatherV2:output:0@sequential/rot_equiv_conv2d_2/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  @*
paddingVALID*
strides
g
%sequential/rot_equiv_conv2d_2/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 RС
%sequential/rot_equiv_conv2d_2/Shape_1Shape,sequential/rot_equiv_pool2d_1/stack:output:0*
T0*
_output_shapes
:*
out_type0	Ж
3sequential/rot_equiv_conv2d_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€И
5sequential/rot_equiv_conv2d_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€
5sequential/rot_equiv_conv2d_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:с
-sequential/rot_equiv_conv2d_2/strided_slice_1StridedSlice.sequential/rot_equiv_conv2d_2/Shape_1:output:0<sequential/rot_equiv_conv2d_2/strided_slice_1/stack:output:0>sequential/rot_equiv_conv2d_2/strided_slice_1/stack_1:output:0>sequential/rot_equiv_conv2d_2/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskg
%sequential/rot_equiv_conv2d_2/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R≥
#sequential/rot_equiv_conv2d_2/sub_1Sub6sequential/rot_equiv_conv2d_2/strided_slice_1:output:0.sequential/rot_equiv_conv2d_2/sub_1/y:output:0*
T0	*
_output_shapes
: Ї
5sequential/rot_equiv_conv2d_2/clip_by_value_1/MinimumMinimum.sequential/rot_equiv_conv2d_2/Const_1:output:0'sequential/rot_equiv_conv2d_2/sub_1:z:0*
T0	*
_output_shapes
: q
/sequential/rot_equiv_conv2d_2/clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ќ
-sequential/rot_equiv_conv2d_2/clip_by_value_1Maximum9sequential/rot_equiv_conv2d_2/clip_by_value_1/Minimum:z:08sequential/rot_equiv_conv2d_2/clip_by_value_1/y:output:0*
T0	*
_output_shapes
: x
-sequential/rot_equiv_conv2d_2/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€™
(sequential/rot_equiv_conv2d_2/GatherV2_1GatherV2,sequential/rot_equiv_pool2d_1/stack:output:01sequential/rot_equiv_conv2d_2/clip_by_value_1:z:06sequential/rot_equiv_conv2d_2/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€"" ƒ
:sequential/rot_equiv_conv2d_2/convolution_1/ReadVariableOpReadVariableOpAsequential_rot_equiv_conv2d_2_convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0П
+sequential/rot_equiv_conv2d_2/convolution_1Conv2D1sequential/rot_equiv_conv2d_2/GatherV2_1:output:0Bsequential/rot_equiv_conv2d_2/convolution_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  @*
paddingVALID*
strides
g
%sequential/rot_equiv_conv2d_2/Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 RС
%sequential/rot_equiv_conv2d_2/Shape_2Shape,sequential/rot_equiv_pool2d_1/stack:output:0*
T0*
_output_shapes
:*
out_type0	Ж
3sequential/rot_equiv_conv2d_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€И
5sequential/rot_equiv_conv2d_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€
5sequential/rot_equiv_conv2d_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:с
-sequential/rot_equiv_conv2d_2/strided_slice_2StridedSlice.sequential/rot_equiv_conv2d_2/Shape_2:output:0<sequential/rot_equiv_conv2d_2/strided_slice_2/stack:output:0>sequential/rot_equiv_conv2d_2/strided_slice_2/stack_1:output:0>sequential/rot_equiv_conv2d_2/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskg
%sequential/rot_equiv_conv2d_2/sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R≥
#sequential/rot_equiv_conv2d_2/sub_2Sub6sequential/rot_equiv_conv2d_2/strided_slice_2:output:0.sequential/rot_equiv_conv2d_2/sub_2/y:output:0*
T0	*
_output_shapes
: Ї
5sequential/rot_equiv_conv2d_2/clip_by_value_2/MinimumMinimum.sequential/rot_equiv_conv2d_2/Const_2:output:0'sequential/rot_equiv_conv2d_2/sub_2:z:0*
T0	*
_output_shapes
: q
/sequential/rot_equiv_conv2d_2/clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ќ
-sequential/rot_equiv_conv2d_2/clip_by_value_2Maximum9sequential/rot_equiv_conv2d_2/clip_by_value_2/Minimum:z:08sequential/rot_equiv_conv2d_2/clip_by_value_2/y:output:0*
T0	*
_output_shapes
: x
-sequential/rot_equiv_conv2d_2/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€™
(sequential/rot_equiv_conv2d_2/GatherV2_2GatherV2,sequential/rot_equiv_pool2d_1/stack:output:01sequential/rot_equiv_conv2d_2/clip_by_value_2:z:06sequential/rot_equiv_conv2d_2/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€"" ƒ
:sequential/rot_equiv_conv2d_2/convolution_2/ReadVariableOpReadVariableOpAsequential_rot_equiv_conv2d_2_convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0П
+sequential/rot_equiv_conv2d_2/convolution_2Conv2D1sequential/rot_equiv_conv2d_2/GatherV2_2:output:0Bsequential/rot_equiv_conv2d_2/convolution_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  @*
paddingVALID*
strides
g
%sequential/rot_equiv_conv2d_2/Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 RС
%sequential/rot_equiv_conv2d_2/Shape_3Shape,sequential/rot_equiv_pool2d_1/stack:output:0*
T0*
_output_shapes
:*
out_type0	Ж
3sequential/rot_equiv_conv2d_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€И
5sequential/rot_equiv_conv2d_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€
5sequential/rot_equiv_conv2d_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:с
-sequential/rot_equiv_conv2d_2/strided_slice_3StridedSlice.sequential/rot_equiv_conv2d_2/Shape_3:output:0<sequential/rot_equiv_conv2d_2/strided_slice_3/stack:output:0>sequential/rot_equiv_conv2d_2/strided_slice_3/stack_1:output:0>sequential/rot_equiv_conv2d_2/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskg
%sequential/rot_equiv_conv2d_2/sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R≥
#sequential/rot_equiv_conv2d_2/sub_3Sub6sequential/rot_equiv_conv2d_2/strided_slice_3:output:0.sequential/rot_equiv_conv2d_2/sub_3/y:output:0*
T0	*
_output_shapes
: Ї
5sequential/rot_equiv_conv2d_2/clip_by_value_3/MinimumMinimum.sequential/rot_equiv_conv2d_2/Const_3:output:0'sequential/rot_equiv_conv2d_2/sub_3:z:0*
T0	*
_output_shapes
: q
/sequential/rot_equiv_conv2d_2/clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ќ
-sequential/rot_equiv_conv2d_2/clip_by_value_3Maximum9sequential/rot_equiv_conv2d_2/clip_by_value_3/Minimum:z:08sequential/rot_equiv_conv2d_2/clip_by_value_3/y:output:0*
T0	*
_output_shapes
: x
-sequential/rot_equiv_conv2d_2/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€™
(sequential/rot_equiv_conv2d_2/GatherV2_3GatherV2,sequential/rot_equiv_pool2d_1/stack:output:01sequential/rot_equiv_conv2d_2/clip_by_value_3:z:06sequential/rot_equiv_conv2d_2/GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€"" ƒ
:sequential/rot_equiv_conv2d_2/convolution_3/ReadVariableOpReadVariableOpAsequential_rot_equiv_conv2d_2_convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0П
+sequential/rot_equiv_conv2d_2/convolution_3Conv2D1sequential/rot_equiv_conv2d_2/GatherV2_3:output:0Bsequential/rot_equiv_conv2d_2/convolution_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  @*
paddingVALID*
strides
Ё
#sequential/rot_equiv_conv2d_2/stackPack2sequential/rot_equiv_conv2d_2/convolution:output:04sequential/rot_equiv_conv2d_2/convolution_1:output:04sequential/rot_equiv_conv2d_2/convolution_2:output:04sequential/rot_equiv_conv2d_2/convolution_3:output:0*
N*
T0*3
_output_shapes!
:€€€€€€€€€  @*
axisю€€€€€€€€Ц
"sequential/rot_equiv_conv2d_2/ReluRelu,sequential/rot_equiv_conv2d_2/stack:output:0*
T0*3
_output_shapes!
:€€€€€€€€€  @Ѓ
4sequential/rot_equiv_conv2d_2/BiasAdd/ReadVariableOpReadVariableOp=sequential_rot_equiv_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ё
%sequential/rot_equiv_conv2d_2/BiasAddBiasAdd0sequential/rot_equiv_conv2d_2/Relu:activations:0<sequential/rot_equiv_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€  @e
#sequential/rot_equiv_pool2d_2/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R С
#sequential/rot_equiv_pool2d_2/ShapeShape.sequential/rot_equiv_conv2d_2/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	Д
1sequential/rot_equiv_pool2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€Ж
3sequential/rot_equiv_pool2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€}
3sequential/rot_equiv_pool2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:з
+sequential/rot_equiv_pool2d_2/strided_sliceStridedSlice,sequential/rot_equiv_pool2d_2/Shape:output:0:sequential/rot_equiv_pool2d_2/strided_slice/stack:output:0<sequential/rot_equiv_pool2d_2/strided_slice/stack_1:output:0<sequential/rot_equiv_pool2d_2/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maske
#sequential/rot_equiv_pool2d_2/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R≠
!sequential/rot_equiv_pool2d_2/subSub4sequential/rot_equiv_pool2d_2/strided_slice:output:0,sequential/rot_equiv_pool2d_2/sub/y:output:0*
T0	*
_output_shapes
: і
3sequential/rot_equiv_pool2d_2/clip_by_value/MinimumMinimum,sequential/rot_equiv_pool2d_2/Const:output:0%sequential/rot_equiv_pool2d_2/sub:z:0*
T0	*
_output_shapes
: o
-sequential/rot_equiv_pool2d_2/clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R »
+sequential/rot_equiv_pool2d_2/clip_by_valueMaximum7sequential/rot_equiv_pool2d_2/clip_by_value/Minimum:z:06sequential/rot_equiv_pool2d_2/clip_by_value/y:output:0*
T0	*
_output_shapes
: v
+sequential/rot_equiv_pool2d_2/GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€¶
&sequential/rot_equiv_pool2d_2/GatherV2GatherV2.sequential/rot_equiv_conv2d_2/BiasAdd:output:0/sequential/rot_equiv_pool2d_2/clip_by_value:z:04sequential/rot_equiv_pool2d_2/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€  @ё
5sequential/rot_equiv_pool2d_2/max_pooling2d_2/MaxPoolMaxPool/sequential/rot_equiv_pool2d_2/GatherV2:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
g
%sequential/rot_equiv_pool2d_2/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 RУ
%sequential/rot_equiv_pool2d_2/Shape_1Shape.sequential/rot_equiv_conv2d_2/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	Ж
3sequential/rot_equiv_pool2d_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€И
5sequential/rot_equiv_pool2d_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€
5sequential/rot_equiv_pool2d_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:с
-sequential/rot_equiv_pool2d_2/strided_slice_1StridedSlice.sequential/rot_equiv_pool2d_2/Shape_1:output:0<sequential/rot_equiv_pool2d_2/strided_slice_1/stack:output:0>sequential/rot_equiv_pool2d_2/strided_slice_1/stack_1:output:0>sequential/rot_equiv_pool2d_2/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskg
%sequential/rot_equiv_pool2d_2/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R≥
#sequential/rot_equiv_pool2d_2/sub_1Sub6sequential/rot_equiv_pool2d_2/strided_slice_1:output:0.sequential/rot_equiv_pool2d_2/sub_1/y:output:0*
T0	*
_output_shapes
: Ї
5sequential/rot_equiv_pool2d_2/clip_by_value_1/MinimumMinimum.sequential/rot_equiv_pool2d_2/Const_1:output:0'sequential/rot_equiv_pool2d_2/sub_1:z:0*
T0	*
_output_shapes
: q
/sequential/rot_equiv_pool2d_2/clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ќ
-sequential/rot_equiv_pool2d_2/clip_by_value_1Maximum9sequential/rot_equiv_pool2d_2/clip_by_value_1/Minimum:z:08sequential/rot_equiv_pool2d_2/clip_by_value_1/y:output:0*
T0	*
_output_shapes
: x
-sequential/rot_equiv_pool2d_2/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ђ
(sequential/rot_equiv_pool2d_2/GatherV2_1GatherV2.sequential/rot_equiv_conv2d_2/BiasAdd:output:01sequential/rot_equiv_pool2d_2/clip_by_value_1:z:06sequential/rot_equiv_pool2d_2/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€  @в
7sequential/rot_equiv_pool2d_2/max_pooling2d_2/MaxPool_1MaxPool1sequential/rot_equiv_pool2d_2/GatherV2_1:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
g
%sequential/rot_equiv_pool2d_2/Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 RУ
%sequential/rot_equiv_pool2d_2/Shape_2Shape.sequential/rot_equiv_conv2d_2/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	Ж
3sequential/rot_equiv_pool2d_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€И
5sequential/rot_equiv_pool2d_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€
5sequential/rot_equiv_pool2d_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:с
-sequential/rot_equiv_pool2d_2/strided_slice_2StridedSlice.sequential/rot_equiv_pool2d_2/Shape_2:output:0<sequential/rot_equiv_pool2d_2/strided_slice_2/stack:output:0>sequential/rot_equiv_pool2d_2/strided_slice_2/stack_1:output:0>sequential/rot_equiv_pool2d_2/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskg
%sequential/rot_equiv_pool2d_2/sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R≥
#sequential/rot_equiv_pool2d_2/sub_2Sub6sequential/rot_equiv_pool2d_2/strided_slice_2:output:0.sequential/rot_equiv_pool2d_2/sub_2/y:output:0*
T0	*
_output_shapes
: Ї
5sequential/rot_equiv_pool2d_2/clip_by_value_2/MinimumMinimum.sequential/rot_equiv_pool2d_2/Const_2:output:0'sequential/rot_equiv_pool2d_2/sub_2:z:0*
T0	*
_output_shapes
: q
/sequential/rot_equiv_pool2d_2/clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ќ
-sequential/rot_equiv_pool2d_2/clip_by_value_2Maximum9sequential/rot_equiv_pool2d_2/clip_by_value_2/Minimum:z:08sequential/rot_equiv_pool2d_2/clip_by_value_2/y:output:0*
T0	*
_output_shapes
: x
-sequential/rot_equiv_pool2d_2/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ђ
(sequential/rot_equiv_pool2d_2/GatherV2_2GatherV2.sequential/rot_equiv_conv2d_2/BiasAdd:output:01sequential/rot_equiv_pool2d_2/clip_by_value_2:z:06sequential/rot_equiv_pool2d_2/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€  @в
7sequential/rot_equiv_pool2d_2/max_pooling2d_2/MaxPool_2MaxPool1sequential/rot_equiv_pool2d_2/GatherV2_2:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
g
%sequential/rot_equiv_pool2d_2/Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 RУ
%sequential/rot_equiv_pool2d_2/Shape_3Shape.sequential/rot_equiv_conv2d_2/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	Ж
3sequential/rot_equiv_pool2d_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€И
5sequential/rot_equiv_pool2d_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€
5sequential/rot_equiv_pool2d_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:с
-sequential/rot_equiv_pool2d_2/strided_slice_3StridedSlice.sequential/rot_equiv_pool2d_2/Shape_3:output:0<sequential/rot_equiv_pool2d_2/strided_slice_3/stack:output:0>sequential/rot_equiv_pool2d_2/strided_slice_3/stack_1:output:0>sequential/rot_equiv_pool2d_2/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskg
%sequential/rot_equiv_pool2d_2/sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R≥
#sequential/rot_equiv_pool2d_2/sub_3Sub6sequential/rot_equiv_pool2d_2/strided_slice_3:output:0.sequential/rot_equiv_pool2d_2/sub_3/y:output:0*
T0	*
_output_shapes
: Ї
5sequential/rot_equiv_pool2d_2/clip_by_value_3/MinimumMinimum.sequential/rot_equiv_pool2d_2/Const_3:output:0'sequential/rot_equiv_pool2d_2/sub_3:z:0*
T0	*
_output_shapes
: q
/sequential/rot_equiv_pool2d_2/clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ќ
-sequential/rot_equiv_pool2d_2/clip_by_value_3Maximum9sequential/rot_equiv_pool2d_2/clip_by_value_3/Minimum:z:08sequential/rot_equiv_pool2d_2/clip_by_value_3/y:output:0*
T0	*
_output_shapes
: x
-sequential/rot_equiv_pool2d_2/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ђ
(sequential/rot_equiv_pool2d_2/GatherV2_3GatherV2.sequential/rot_equiv_conv2d_2/BiasAdd:output:01sequential/rot_equiv_pool2d_2/clip_by_value_3:z:06sequential/rot_equiv_pool2d_2/GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€  @в
7sequential/rot_equiv_pool2d_2/max_pooling2d_2/MaxPool_3MaxPool1sequential/rot_equiv_pool2d_2/GatherV2_3:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
Н
#sequential/rot_equiv_pool2d_2/stackPack>sequential/rot_equiv_pool2d_2/max_pooling2d_2/MaxPool:output:0@sequential/rot_equiv_pool2d_2/max_pooling2d_2/MaxPool_1:output:0@sequential/rot_equiv_pool2d_2/max_pooling2d_2/MaxPool_2:output:0@sequential/rot_equiv_pool2d_2/max_pooling2d_2/MaxPool_3:output:0*
N*
T0*3
_output_shapes!
:€€€€€€€€€@*
axisю€€€€€€€€e
#sequential/rot_equiv_conv2d_3/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R П
#sequential/rot_equiv_conv2d_3/ShapeShape,sequential/rot_equiv_pool2d_2/stack:output:0*
T0*
_output_shapes
:*
out_type0	Д
1sequential/rot_equiv_conv2d_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€Ж
3sequential/rot_equiv_conv2d_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€}
3sequential/rot_equiv_conv2d_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:з
+sequential/rot_equiv_conv2d_3/strided_sliceStridedSlice,sequential/rot_equiv_conv2d_3/Shape:output:0:sequential/rot_equiv_conv2d_3/strided_slice/stack:output:0<sequential/rot_equiv_conv2d_3/strided_slice/stack_1:output:0<sequential/rot_equiv_conv2d_3/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maske
#sequential/rot_equiv_conv2d_3/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R≠
!sequential/rot_equiv_conv2d_3/subSub4sequential/rot_equiv_conv2d_3/strided_slice:output:0,sequential/rot_equiv_conv2d_3/sub/y:output:0*
T0	*
_output_shapes
: і
3sequential/rot_equiv_conv2d_3/clip_by_value/MinimumMinimum,sequential/rot_equiv_conv2d_3/Const:output:0%sequential/rot_equiv_conv2d_3/sub:z:0*
T0	*
_output_shapes
: o
-sequential/rot_equiv_conv2d_3/clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R »
+sequential/rot_equiv_conv2d_3/clip_by_valueMaximum7sequential/rot_equiv_conv2d_3/clip_by_value/Minimum:z:06sequential/rot_equiv_conv2d_3/clip_by_value/y:output:0*
T0	*
_output_shapes
: v
+sequential/rot_equiv_conv2d_3/GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€§
&sequential/rot_equiv_conv2d_3/GatherV2GatherV2,sequential/rot_equiv_pool2d_2/stack:output:0/sequential/rot_equiv_conv2d_3/clip_by_value:z:04sequential/rot_equiv_conv2d_3/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@¬
8sequential/rot_equiv_conv2d_3/convolution/ReadVariableOpReadVariableOpAsequential_rot_equiv_conv2d_3_convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype0Й
)sequential/rot_equiv_conv2d_3/convolutionConv2D/sequential/rot_equiv_conv2d_3/GatherV2:output:0@sequential/rot_equiv_conv2d_3/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingVALID*
strides
g
%sequential/rot_equiv_conv2d_3/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 RС
%sequential/rot_equiv_conv2d_3/Shape_1Shape,sequential/rot_equiv_pool2d_2/stack:output:0*
T0*
_output_shapes
:*
out_type0	Ж
3sequential/rot_equiv_conv2d_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€И
5sequential/rot_equiv_conv2d_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€
5sequential/rot_equiv_conv2d_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:с
-sequential/rot_equiv_conv2d_3/strided_slice_1StridedSlice.sequential/rot_equiv_conv2d_3/Shape_1:output:0<sequential/rot_equiv_conv2d_3/strided_slice_1/stack:output:0>sequential/rot_equiv_conv2d_3/strided_slice_1/stack_1:output:0>sequential/rot_equiv_conv2d_3/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskg
%sequential/rot_equiv_conv2d_3/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R≥
#sequential/rot_equiv_conv2d_3/sub_1Sub6sequential/rot_equiv_conv2d_3/strided_slice_1:output:0.sequential/rot_equiv_conv2d_3/sub_1/y:output:0*
T0	*
_output_shapes
: Ї
5sequential/rot_equiv_conv2d_3/clip_by_value_1/MinimumMinimum.sequential/rot_equiv_conv2d_3/Const_1:output:0'sequential/rot_equiv_conv2d_3/sub_1:z:0*
T0	*
_output_shapes
: q
/sequential/rot_equiv_conv2d_3/clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ќ
-sequential/rot_equiv_conv2d_3/clip_by_value_1Maximum9sequential/rot_equiv_conv2d_3/clip_by_value_1/Minimum:z:08sequential/rot_equiv_conv2d_3/clip_by_value_1/y:output:0*
T0	*
_output_shapes
: x
-sequential/rot_equiv_conv2d_3/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€™
(sequential/rot_equiv_conv2d_3/GatherV2_1GatherV2,sequential/rot_equiv_pool2d_2/stack:output:01sequential/rot_equiv_conv2d_3/clip_by_value_1:z:06sequential/rot_equiv_conv2d_3/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@ƒ
:sequential/rot_equiv_conv2d_3/convolution_1/ReadVariableOpReadVariableOpAsequential_rot_equiv_conv2d_3_convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype0П
+sequential/rot_equiv_conv2d_3/convolution_1Conv2D1sequential/rot_equiv_conv2d_3/GatherV2_1:output:0Bsequential/rot_equiv_conv2d_3/convolution_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingVALID*
strides
g
%sequential/rot_equiv_conv2d_3/Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 RС
%sequential/rot_equiv_conv2d_3/Shape_2Shape,sequential/rot_equiv_pool2d_2/stack:output:0*
T0*
_output_shapes
:*
out_type0	Ж
3sequential/rot_equiv_conv2d_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€И
5sequential/rot_equiv_conv2d_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€
5sequential/rot_equiv_conv2d_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:с
-sequential/rot_equiv_conv2d_3/strided_slice_2StridedSlice.sequential/rot_equiv_conv2d_3/Shape_2:output:0<sequential/rot_equiv_conv2d_3/strided_slice_2/stack:output:0>sequential/rot_equiv_conv2d_3/strided_slice_2/stack_1:output:0>sequential/rot_equiv_conv2d_3/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskg
%sequential/rot_equiv_conv2d_3/sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R≥
#sequential/rot_equiv_conv2d_3/sub_2Sub6sequential/rot_equiv_conv2d_3/strided_slice_2:output:0.sequential/rot_equiv_conv2d_3/sub_2/y:output:0*
T0	*
_output_shapes
: Ї
5sequential/rot_equiv_conv2d_3/clip_by_value_2/MinimumMinimum.sequential/rot_equiv_conv2d_3/Const_2:output:0'sequential/rot_equiv_conv2d_3/sub_2:z:0*
T0	*
_output_shapes
: q
/sequential/rot_equiv_conv2d_3/clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ќ
-sequential/rot_equiv_conv2d_3/clip_by_value_2Maximum9sequential/rot_equiv_conv2d_3/clip_by_value_2/Minimum:z:08sequential/rot_equiv_conv2d_3/clip_by_value_2/y:output:0*
T0	*
_output_shapes
: x
-sequential/rot_equiv_conv2d_3/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€™
(sequential/rot_equiv_conv2d_3/GatherV2_2GatherV2,sequential/rot_equiv_pool2d_2/stack:output:01sequential/rot_equiv_conv2d_3/clip_by_value_2:z:06sequential/rot_equiv_conv2d_3/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@ƒ
:sequential/rot_equiv_conv2d_3/convolution_2/ReadVariableOpReadVariableOpAsequential_rot_equiv_conv2d_3_convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype0П
+sequential/rot_equiv_conv2d_3/convolution_2Conv2D1sequential/rot_equiv_conv2d_3/GatherV2_2:output:0Bsequential/rot_equiv_conv2d_3/convolution_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingVALID*
strides
g
%sequential/rot_equiv_conv2d_3/Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 RС
%sequential/rot_equiv_conv2d_3/Shape_3Shape,sequential/rot_equiv_pool2d_2/stack:output:0*
T0*
_output_shapes
:*
out_type0	Ж
3sequential/rot_equiv_conv2d_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€И
5sequential/rot_equiv_conv2d_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€
5sequential/rot_equiv_conv2d_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:с
-sequential/rot_equiv_conv2d_3/strided_slice_3StridedSlice.sequential/rot_equiv_conv2d_3/Shape_3:output:0<sequential/rot_equiv_conv2d_3/strided_slice_3/stack:output:0>sequential/rot_equiv_conv2d_3/strided_slice_3/stack_1:output:0>sequential/rot_equiv_conv2d_3/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskg
%sequential/rot_equiv_conv2d_3/sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R≥
#sequential/rot_equiv_conv2d_3/sub_3Sub6sequential/rot_equiv_conv2d_3/strided_slice_3:output:0.sequential/rot_equiv_conv2d_3/sub_3/y:output:0*
T0	*
_output_shapes
: Ї
5sequential/rot_equiv_conv2d_3/clip_by_value_3/MinimumMinimum.sequential/rot_equiv_conv2d_3/Const_3:output:0'sequential/rot_equiv_conv2d_3/sub_3:z:0*
T0	*
_output_shapes
: q
/sequential/rot_equiv_conv2d_3/clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ќ
-sequential/rot_equiv_conv2d_3/clip_by_value_3Maximum9sequential/rot_equiv_conv2d_3/clip_by_value_3/Minimum:z:08sequential/rot_equiv_conv2d_3/clip_by_value_3/y:output:0*
T0	*
_output_shapes
: x
-sequential/rot_equiv_conv2d_3/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€™
(sequential/rot_equiv_conv2d_3/GatherV2_3GatherV2,sequential/rot_equiv_pool2d_2/stack:output:01sequential/rot_equiv_conv2d_3/clip_by_value_3:z:06sequential/rot_equiv_conv2d_3/GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@ƒ
:sequential/rot_equiv_conv2d_3/convolution_3/ReadVariableOpReadVariableOpAsequential_rot_equiv_conv2d_3_convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype0П
+sequential/rot_equiv_conv2d_3/convolution_3Conv2D1sequential/rot_equiv_conv2d_3/GatherV2_3:output:0Bsequential/rot_equiv_conv2d_3/convolution_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingVALID*
strides
Ё
#sequential/rot_equiv_conv2d_3/stackPack2sequential/rot_equiv_conv2d_3/convolution:output:04sequential/rot_equiv_conv2d_3/convolution_1:output:04sequential/rot_equiv_conv2d_3/convolution_2:output:04sequential/rot_equiv_conv2d_3/convolution_3:output:0*
N*
T0*3
_output_shapes!
:€€€€€€€€€@*
axisю€€€€€€€€Ц
"sequential/rot_equiv_conv2d_3/ReluRelu,sequential/rot_equiv_conv2d_3/stack:output:0*
T0*3
_output_shapes!
:€€€€€€€€€@Ѓ
4sequential/rot_equiv_conv2d_3/BiasAdd/ReadVariableOpReadVariableOp=sequential_rot_equiv_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ё
%sequential/rot_equiv_conv2d_3/BiasAddBiasAdd0sequential/rot_equiv_conv2d_3/Relu:activations:0<sequential/rot_equiv_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€@e
#sequential/rot_equiv_pool2d_3/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R С
#sequential/rot_equiv_pool2d_3/ShapeShape.sequential/rot_equiv_conv2d_3/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	Д
1sequential/rot_equiv_pool2d_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€Ж
3sequential/rot_equiv_pool2d_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€}
3sequential/rot_equiv_pool2d_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:з
+sequential/rot_equiv_pool2d_3/strided_sliceStridedSlice,sequential/rot_equiv_pool2d_3/Shape:output:0:sequential/rot_equiv_pool2d_3/strided_slice/stack:output:0<sequential/rot_equiv_pool2d_3/strided_slice/stack_1:output:0<sequential/rot_equiv_pool2d_3/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maske
#sequential/rot_equiv_pool2d_3/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R≠
!sequential/rot_equiv_pool2d_3/subSub4sequential/rot_equiv_pool2d_3/strided_slice:output:0,sequential/rot_equiv_pool2d_3/sub/y:output:0*
T0	*
_output_shapes
: і
3sequential/rot_equiv_pool2d_3/clip_by_value/MinimumMinimum,sequential/rot_equiv_pool2d_3/Const:output:0%sequential/rot_equiv_pool2d_3/sub:z:0*
T0	*
_output_shapes
: o
-sequential/rot_equiv_pool2d_3/clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R »
+sequential/rot_equiv_pool2d_3/clip_by_valueMaximum7sequential/rot_equiv_pool2d_3/clip_by_value/Minimum:z:06sequential/rot_equiv_pool2d_3/clip_by_value/y:output:0*
T0	*
_output_shapes
: v
+sequential/rot_equiv_pool2d_3/GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€¶
&sequential/rot_equiv_pool2d_3/GatherV2GatherV2.sequential/rot_equiv_conv2d_3/BiasAdd:output:0/sequential/rot_equiv_pool2d_3/clip_by_value:z:04sequential/rot_equiv_pool2d_3/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@ё
5sequential/rot_equiv_pool2d_3/max_pooling2d_3/MaxPoolMaxPool/sequential/rot_equiv_pool2d_3/GatherV2:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
g
%sequential/rot_equiv_pool2d_3/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 RУ
%sequential/rot_equiv_pool2d_3/Shape_1Shape.sequential/rot_equiv_conv2d_3/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	Ж
3sequential/rot_equiv_pool2d_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€И
5sequential/rot_equiv_pool2d_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€
5sequential/rot_equiv_pool2d_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:с
-sequential/rot_equiv_pool2d_3/strided_slice_1StridedSlice.sequential/rot_equiv_pool2d_3/Shape_1:output:0<sequential/rot_equiv_pool2d_3/strided_slice_1/stack:output:0>sequential/rot_equiv_pool2d_3/strided_slice_1/stack_1:output:0>sequential/rot_equiv_pool2d_3/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskg
%sequential/rot_equiv_pool2d_3/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R≥
#sequential/rot_equiv_pool2d_3/sub_1Sub6sequential/rot_equiv_pool2d_3/strided_slice_1:output:0.sequential/rot_equiv_pool2d_3/sub_1/y:output:0*
T0	*
_output_shapes
: Ї
5sequential/rot_equiv_pool2d_3/clip_by_value_1/MinimumMinimum.sequential/rot_equiv_pool2d_3/Const_1:output:0'sequential/rot_equiv_pool2d_3/sub_1:z:0*
T0	*
_output_shapes
: q
/sequential/rot_equiv_pool2d_3/clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ќ
-sequential/rot_equiv_pool2d_3/clip_by_value_1Maximum9sequential/rot_equiv_pool2d_3/clip_by_value_1/Minimum:z:08sequential/rot_equiv_pool2d_3/clip_by_value_1/y:output:0*
T0	*
_output_shapes
: x
-sequential/rot_equiv_pool2d_3/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ђ
(sequential/rot_equiv_pool2d_3/GatherV2_1GatherV2.sequential/rot_equiv_conv2d_3/BiasAdd:output:01sequential/rot_equiv_pool2d_3/clip_by_value_1:z:06sequential/rot_equiv_pool2d_3/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@в
7sequential/rot_equiv_pool2d_3/max_pooling2d_3/MaxPool_1MaxPool1sequential/rot_equiv_pool2d_3/GatherV2_1:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
g
%sequential/rot_equiv_pool2d_3/Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 RУ
%sequential/rot_equiv_pool2d_3/Shape_2Shape.sequential/rot_equiv_conv2d_3/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	Ж
3sequential/rot_equiv_pool2d_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€И
5sequential/rot_equiv_pool2d_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€
5sequential/rot_equiv_pool2d_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:с
-sequential/rot_equiv_pool2d_3/strided_slice_2StridedSlice.sequential/rot_equiv_pool2d_3/Shape_2:output:0<sequential/rot_equiv_pool2d_3/strided_slice_2/stack:output:0>sequential/rot_equiv_pool2d_3/strided_slice_2/stack_1:output:0>sequential/rot_equiv_pool2d_3/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskg
%sequential/rot_equiv_pool2d_3/sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R≥
#sequential/rot_equiv_pool2d_3/sub_2Sub6sequential/rot_equiv_pool2d_3/strided_slice_2:output:0.sequential/rot_equiv_pool2d_3/sub_2/y:output:0*
T0	*
_output_shapes
: Ї
5sequential/rot_equiv_pool2d_3/clip_by_value_2/MinimumMinimum.sequential/rot_equiv_pool2d_3/Const_2:output:0'sequential/rot_equiv_pool2d_3/sub_2:z:0*
T0	*
_output_shapes
: q
/sequential/rot_equiv_pool2d_3/clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ќ
-sequential/rot_equiv_pool2d_3/clip_by_value_2Maximum9sequential/rot_equiv_pool2d_3/clip_by_value_2/Minimum:z:08sequential/rot_equiv_pool2d_3/clip_by_value_2/y:output:0*
T0	*
_output_shapes
: x
-sequential/rot_equiv_pool2d_3/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ђ
(sequential/rot_equiv_pool2d_3/GatherV2_2GatherV2.sequential/rot_equiv_conv2d_3/BiasAdd:output:01sequential/rot_equiv_pool2d_3/clip_by_value_2:z:06sequential/rot_equiv_pool2d_3/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@в
7sequential/rot_equiv_pool2d_3/max_pooling2d_3/MaxPool_2MaxPool1sequential/rot_equiv_pool2d_3/GatherV2_2:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
g
%sequential/rot_equiv_pool2d_3/Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 RУ
%sequential/rot_equiv_pool2d_3/Shape_3Shape.sequential/rot_equiv_conv2d_3/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	Ж
3sequential/rot_equiv_pool2d_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€И
5sequential/rot_equiv_pool2d_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€
5sequential/rot_equiv_pool2d_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:с
-sequential/rot_equiv_pool2d_3/strided_slice_3StridedSlice.sequential/rot_equiv_pool2d_3/Shape_3:output:0<sequential/rot_equiv_pool2d_3/strided_slice_3/stack:output:0>sequential/rot_equiv_pool2d_3/strided_slice_3/stack_1:output:0>sequential/rot_equiv_pool2d_3/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskg
%sequential/rot_equiv_pool2d_3/sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R≥
#sequential/rot_equiv_pool2d_3/sub_3Sub6sequential/rot_equiv_pool2d_3/strided_slice_3:output:0.sequential/rot_equiv_pool2d_3/sub_3/y:output:0*
T0	*
_output_shapes
: Ї
5sequential/rot_equiv_pool2d_3/clip_by_value_3/MinimumMinimum.sequential/rot_equiv_pool2d_3/Const_3:output:0'sequential/rot_equiv_pool2d_3/sub_3:z:0*
T0	*
_output_shapes
: q
/sequential/rot_equiv_pool2d_3/clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ќ
-sequential/rot_equiv_pool2d_3/clip_by_value_3Maximum9sequential/rot_equiv_pool2d_3/clip_by_value_3/Minimum:z:08sequential/rot_equiv_pool2d_3/clip_by_value_3/y:output:0*
T0	*
_output_shapes
: x
-sequential/rot_equiv_pool2d_3/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ђ
(sequential/rot_equiv_pool2d_3/GatherV2_3GatherV2.sequential/rot_equiv_conv2d_3/BiasAdd:output:01sequential/rot_equiv_pool2d_3/clip_by_value_3:z:06sequential/rot_equiv_pool2d_3/GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@в
7sequential/rot_equiv_pool2d_3/max_pooling2d_3/MaxPool_3MaxPool1sequential/rot_equiv_pool2d_3/GatherV2_3:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
Н
#sequential/rot_equiv_pool2d_3/stackPack>sequential/rot_equiv_pool2d_3/max_pooling2d_3/MaxPool:output:0@sequential/rot_equiv_pool2d_3/max_pooling2d_3/MaxPool_1:output:0@sequential/rot_equiv_pool2d_3/max_pooling2d_3/MaxPool_2:output:0@sequential/rot_equiv_pool2d_3/max_pooling2d_3/MaxPool_3:output:0*
N*
T0*3
_output_shapes!
:€€€€€€€€€@*
axisю€€€€€€€€e
#sequential/rot_equiv_conv2d_4/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R П
#sequential/rot_equiv_conv2d_4/ShapeShape,sequential/rot_equiv_pool2d_3/stack:output:0*
T0*
_output_shapes
:*
out_type0	Д
1sequential/rot_equiv_conv2d_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€Ж
3sequential/rot_equiv_conv2d_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€}
3sequential/rot_equiv_conv2d_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:з
+sequential/rot_equiv_conv2d_4/strided_sliceStridedSlice,sequential/rot_equiv_conv2d_4/Shape:output:0:sequential/rot_equiv_conv2d_4/strided_slice/stack:output:0<sequential/rot_equiv_conv2d_4/strided_slice/stack_1:output:0<sequential/rot_equiv_conv2d_4/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maske
#sequential/rot_equiv_conv2d_4/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R≠
!sequential/rot_equiv_conv2d_4/subSub4sequential/rot_equiv_conv2d_4/strided_slice:output:0,sequential/rot_equiv_conv2d_4/sub/y:output:0*
T0	*
_output_shapes
: і
3sequential/rot_equiv_conv2d_4/clip_by_value/MinimumMinimum,sequential/rot_equiv_conv2d_4/Const:output:0%sequential/rot_equiv_conv2d_4/sub:z:0*
T0	*
_output_shapes
: o
-sequential/rot_equiv_conv2d_4/clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R »
+sequential/rot_equiv_conv2d_4/clip_by_valueMaximum7sequential/rot_equiv_conv2d_4/clip_by_value/Minimum:z:06sequential/rot_equiv_conv2d_4/clip_by_value/y:output:0*
T0	*
_output_shapes
: v
+sequential/rot_equiv_conv2d_4/GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€§
&sequential/rot_equiv_conv2d_4/GatherV2GatherV2,sequential/rot_equiv_pool2d_3/stack:output:0/sequential/rot_equiv_conv2d_4/clip_by_value:z:04sequential/rot_equiv_conv2d_4/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@√
8sequential/rot_equiv_conv2d_4/convolution/ReadVariableOpReadVariableOpAsequential_rot_equiv_conv2d_4_convolution_readvariableop_resource*'
_output_shapes
:@А*
dtype0К
)sequential/rot_equiv_conv2d_4/convolutionConv2D/sequential/rot_equiv_conv2d_4/GatherV2:output:0@sequential/rot_equiv_conv2d_4/convolution/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingVALID*
strides
g
%sequential/rot_equiv_conv2d_4/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 RС
%sequential/rot_equiv_conv2d_4/Shape_1Shape,sequential/rot_equiv_pool2d_3/stack:output:0*
T0*
_output_shapes
:*
out_type0	Ж
3sequential/rot_equiv_conv2d_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€И
5sequential/rot_equiv_conv2d_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€
5sequential/rot_equiv_conv2d_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:с
-sequential/rot_equiv_conv2d_4/strided_slice_1StridedSlice.sequential/rot_equiv_conv2d_4/Shape_1:output:0<sequential/rot_equiv_conv2d_4/strided_slice_1/stack:output:0>sequential/rot_equiv_conv2d_4/strided_slice_1/stack_1:output:0>sequential/rot_equiv_conv2d_4/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskg
%sequential/rot_equiv_conv2d_4/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R≥
#sequential/rot_equiv_conv2d_4/sub_1Sub6sequential/rot_equiv_conv2d_4/strided_slice_1:output:0.sequential/rot_equiv_conv2d_4/sub_1/y:output:0*
T0	*
_output_shapes
: Ї
5sequential/rot_equiv_conv2d_4/clip_by_value_1/MinimumMinimum.sequential/rot_equiv_conv2d_4/Const_1:output:0'sequential/rot_equiv_conv2d_4/sub_1:z:0*
T0	*
_output_shapes
: q
/sequential/rot_equiv_conv2d_4/clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ќ
-sequential/rot_equiv_conv2d_4/clip_by_value_1Maximum9sequential/rot_equiv_conv2d_4/clip_by_value_1/Minimum:z:08sequential/rot_equiv_conv2d_4/clip_by_value_1/y:output:0*
T0	*
_output_shapes
: x
-sequential/rot_equiv_conv2d_4/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€™
(sequential/rot_equiv_conv2d_4/GatherV2_1GatherV2,sequential/rot_equiv_pool2d_3/stack:output:01sequential/rot_equiv_conv2d_4/clip_by_value_1:z:06sequential/rot_equiv_conv2d_4/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@≈
:sequential/rot_equiv_conv2d_4/convolution_1/ReadVariableOpReadVariableOpAsequential_rot_equiv_conv2d_4_convolution_readvariableop_resource*'
_output_shapes
:@А*
dtype0Р
+sequential/rot_equiv_conv2d_4/convolution_1Conv2D1sequential/rot_equiv_conv2d_4/GatherV2_1:output:0Bsequential/rot_equiv_conv2d_4/convolution_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingVALID*
strides
g
%sequential/rot_equiv_conv2d_4/Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 RС
%sequential/rot_equiv_conv2d_4/Shape_2Shape,sequential/rot_equiv_pool2d_3/stack:output:0*
T0*
_output_shapes
:*
out_type0	Ж
3sequential/rot_equiv_conv2d_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€И
5sequential/rot_equiv_conv2d_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€
5sequential/rot_equiv_conv2d_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:с
-sequential/rot_equiv_conv2d_4/strided_slice_2StridedSlice.sequential/rot_equiv_conv2d_4/Shape_2:output:0<sequential/rot_equiv_conv2d_4/strided_slice_2/stack:output:0>sequential/rot_equiv_conv2d_4/strided_slice_2/stack_1:output:0>sequential/rot_equiv_conv2d_4/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskg
%sequential/rot_equiv_conv2d_4/sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R≥
#sequential/rot_equiv_conv2d_4/sub_2Sub6sequential/rot_equiv_conv2d_4/strided_slice_2:output:0.sequential/rot_equiv_conv2d_4/sub_2/y:output:0*
T0	*
_output_shapes
: Ї
5sequential/rot_equiv_conv2d_4/clip_by_value_2/MinimumMinimum.sequential/rot_equiv_conv2d_4/Const_2:output:0'sequential/rot_equiv_conv2d_4/sub_2:z:0*
T0	*
_output_shapes
: q
/sequential/rot_equiv_conv2d_4/clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ќ
-sequential/rot_equiv_conv2d_4/clip_by_value_2Maximum9sequential/rot_equiv_conv2d_4/clip_by_value_2/Minimum:z:08sequential/rot_equiv_conv2d_4/clip_by_value_2/y:output:0*
T0	*
_output_shapes
: x
-sequential/rot_equiv_conv2d_4/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€™
(sequential/rot_equiv_conv2d_4/GatherV2_2GatherV2,sequential/rot_equiv_pool2d_3/stack:output:01sequential/rot_equiv_conv2d_4/clip_by_value_2:z:06sequential/rot_equiv_conv2d_4/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@≈
:sequential/rot_equiv_conv2d_4/convolution_2/ReadVariableOpReadVariableOpAsequential_rot_equiv_conv2d_4_convolution_readvariableop_resource*'
_output_shapes
:@А*
dtype0Р
+sequential/rot_equiv_conv2d_4/convolution_2Conv2D1sequential/rot_equiv_conv2d_4/GatherV2_2:output:0Bsequential/rot_equiv_conv2d_4/convolution_2/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingVALID*
strides
g
%sequential/rot_equiv_conv2d_4/Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 RС
%sequential/rot_equiv_conv2d_4/Shape_3Shape,sequential/rot_equiv_pool2d_3/stack:output:0*
T0*
_output_shapes
:*
out_type0	Ж
3sequential/rot_equiv_conv2d_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€И
5sequential/rot_equiv_conv2d_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€
5sequential/rot_equiv_conv2d_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:с
-sequential/rot_equiv_conv2d_4/strided_slice_3StridedSlice.sequential/rot_equiv_conv2d_4/Shape_3:output:0<sequential/rot_equiv_conv2d_4/strided_slice_3/stack:output:0>sequential/rot_equiv_conv2d_4/strided_slice_3/stack_1:output:0>sequential/rot_equiv_conv2d_4/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskg
%sequential/rot_equiv_conv2d_4/sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R≥
#sequential/rot_equiv_conv2d_4/sub_3Sub6sequential/rot_equiv_conv2d_4/strided_slice_3:output:0.sequential/rot_equiv_conv2d_4/sub_3/y:output:0*
T0	*
_output_shapes
: Ї
5sequential/rot_equiv_conv2d_4/clip_by_value_3/MinimumMinimum.sequential/rot_equiv_conv2d_4/Const_3:output:0'sequential/rot_equiv_conv2d_4/sub_3:z:0*
T0	*
_output_shapes
: q
/sequential/rot_equiv_conv2d_4/clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ќ
-sequential/rot_equiv_conv2d_4/clip_by_value_3Maximum9sequential/rot_equiv_conv2d_4/clip_by_value_3/Minimum:z:08sequential/rot_equiv_conv2d_4/clip_by_value_3/y:output:0*
T0	*
_output_shapes
: x
-sequential/rot_equiv_conv2d_4/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€™
(sequential/rot_equiv_conv2d_4/GatherV2_3GatherV2,sequential/rot_equiv_pool2d_3/stack:output:01sequential/rot_equiv_conv2d_4/clip_by_value_3:z:06sequential/rot_equiv_conv2d_4/GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@≈
:sequential/rot_equiv_conv2d_4/convolution_3/ReadVariableOpReadVariableOpAsequential_rot_equiv_conv2d_4_convolution_readvariableop_resource*'
_output_shapes
:@А*
dtype0Р
+sequential/rot_equiv_conv2d_4/convolution_3Conv2D1sequential/rot_equiv_conv2d_4/GatherV2_3:output:0Bsequential/rot_equiv_conv2d_4/convolution_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingVALID*
strides
ё
#sequential/rot_equiv_conv2d_4/stackPack2sequential/rot_equiv_conv2d_4/convolution:output:04sequential/rot_equiv_conv2d_4/convolution_1:output:04sequential/rot_equiv_conv2d_4/convolution_2:output:04sequential/rot_equiv_conv2d_4/convolution_3:output:0*
N*
T0*4
_output_shapes"
 :€€€€€€€€€А*
axisю€€€€€€€€Ч
"sequential/rot_equiv_conv2d_4/ReluRelu,sequential/rot_equiv_conv2d_4/stack:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аѓ
4sequential/rot_equiv_conv2d_4/BiasAdd/ReadVariableOpReadVariableOp=sequential_rot_equiv_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0я
%sequential/rot_equiv_conv2d_4/BiasAddBiasAdd0sequential/rot_equiv_conv2d_4/Relu:activations:0<sequential/rot_equiv_conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аx
-sequential/rot_inv_pool/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€≈
sequential/rot_inv_pool/MaxMax.sequential/rot_equiv_conv2d_4/BiasAdd:output:06sequential/rot_inv_pool/Max/reduction_indices:output:0*
T0*0
_output_shapes
:€€€€€€€€€Аi
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€А  °
sequential/flatten/ReshapeReshape$sequential/rot_inv_pool/Max:output:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€АЧ
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes
:	А *
dtype0®
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Ф
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0©
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ r
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ Ъ
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype0ђ
sequential/dense_1/MatMulMatMul#sequential/dense/Relu:activations:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ш
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ѓ
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
IdentityIdentity#sequential/dense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Н
NoOpNoOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp3^sequential/rot_equiv_conv2d/BiasAdd/ReadVariableOp+^sequential/rot_equiv_conv2d/ReadVariableOp-^sequential/rot_equiv_conv2d/ReadVariableOp_1-^sequential/rot_equiv_conv2d/ReadVariableOp_27^sequential/rot_equiv_conv2d/convolution/ReadVariableOp5^sequential/rot_equiv_conv2d_1/BiasAdd/ReadVariableOp9^sequential/rot_equiv_conv2d_1/convolution/ReadVariableOp;^sequential/rot_equiv_conv2d_1/convolution_1/ReadVariableOp;^sequential/rot_equiv_conv2d_1/convolution_2/ReadVariableOp;^sequential/rot_equiv_conv2d_1/convolution_3/ReadVariableOp5^sequential/rot_equiv_conv2d_2/BiasAdd/ReadVariableOp9^sequential/rot_equiv_conv2d_2/convolution/ReadVariableOp;^sequential/rot_equiv_conv2d_2/convolution_1/ReadVariableOp;^sequential/rot_equiv_conv2d_2/convolution_2/ReadVariableOp;^sequential/rot_equiv_conv2d_2/convolution_3/ReadVariableOp5^sequential/rot_equiv_conv2d_3/BiasAdd/ReadVariableOp9^sequential/rot_equiv_conv2d_3/convolution/ReadVariableOp;^sequential/rot_equiv_conv2d_3/convolution_1/ReadVariableOp;^sequential/rot_equiv_conv2d_3/convolution_2/ReadVariableOp;^sequential/rot_equiv_conv2d_3/convolution_3/ReadVariableOp5^sequential/rot_equiv_conv2d_4/BiasAdd/ReadVariableOp9^sequential/rot_equiv_conv2d_4/convolution/ReadVariableOp;^sequential/rot_equiv_conv2d_4/convolution_1/ReadVariableOp;^sequential/rot_equiv_conv2d_4/convolution_2/ReadVariableOp;^sequential/rot_equiv_conv2d_4/convolution_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:€€€€€€€€€РР: : : : : : : : : : : : : : 2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2h
2sequential/rot_equiv_conv2d/BiasAdd/ReadVariableOp2sequential/rot_equiv_conv2d/BiasAdd/ReadVariableOp2X
*sequential/rot_equiv_conv2d/ReadVariableOp*sequential/rot_equiv_conv2d/ReadVariableOp2\
,sequential/rot_equiv_conv2d/ReadVariableOp_1,sequential/rot_equiv_conv2d/ReadVariableOp_12\
,sequential/rot_equiv_conv2d/ReadVariableOp_2,sequential/rot_equiv_conv2d/ReadVariableOp_22p
6sequential/rot_equiv_conv2d/convolution/ReadVariableOp6sequential/rot_equiv_conv2d/convolution/ReadVariableOp2l
4sequential/rot_equiv_conv2d_1/BiasAdd/ReadVariableOp4sequential/rot_equiv_conv2d_1/BiasAdd/ReadVariableOp2t
8sequential/rot_equiv_conv2d_1/convolution/ReadVariableOp8sequential/rot_equiv_conv2d_1/convolution/ReadVariableOp2x
:sequential/rot_equiv_conv2d_1/convolution_1/ReadVariableOp:sequential/rot_equiv_conv2d_1/convolution_1/ReadVariableOp2x
:sequential/rot_equiv_conv2d_1/convolution_2/ReadVariableOp:sequential/rot_equiv_conv2d_1/convolution_2/ReadVariableOp2x
:sequential/rot_equiv_conv2d_1/convolution_3/ReadVariableOp:sequential/rot_equiv_conv2d_1/convolution_3/ReadVariableOp2l
4sequential/rot_equiv_conv2d_2/BiasAdd/ReadVariableOp4sequential/rot_equiv_conv2d_2/BiasAdd/ReadVariableOp2t
8sequential/rot_equiv_conv2d_2/convolution/ReadVariableOp8sequential/rot_equiv_conv2d_2/convolution/ReadVariableOp2x
:sequential/rot_equiv_conv2d_2/convolution_1/ReadVariableOp:sequential/rot_equiv_conv2d_2/convolution_1/ReadVariableOp2x
:sequential/rot_equiv_conv2d_2/convolution_2/ReadVariableOp:sequential/rot_equiv_conv2d_2/convolution_2/ReadVariableOp2x
:sequential/rot_equiv_conv2d_2/convolution_3/ReadVariableOp:sequential/rot_equiv_conv2d_2/convolution_3/ReadVariableOp2l
4sequential/rot_equiv_conv2d_3/BiasAdd/ReadVariableOp4sequential/rot_equiv_conv2d_3/BiasAdd/ReadVariableOp2t
8sequential/rot_equiv_conv2d_3/convolution/ReadVariableOp8sequential/rot_equiv_conv2d_3/convolution/ReadVariableOp2x
:sequential/rot_equiv_conv2d_3/convolution_1/ReadVariableOp:sequential/rot_equiv_conv2d_3/convolution_1/ReadVariableOp2x
:sequential/rot_equiv_conv2d_3/convolution_2/ReadVariableOp:sequential/rot_equiv_conv2d_3/convolution_2/ReadVariableOp2x
:sequential/rot_equiv_conv2d_3/convolution_3/ReadVariableOp:sequential/rot_equiv_conv2d_3/convolution_3/ReadVariableOp2l
4sequential/rot_equiv_conv2d_4/BiasAdd/ReadVariableOp4sequential/rot_equiv_conv2d_4/BiasAdd/ReadVariableOp2t
8sequential/rot_equiv_conv2d_4/convolution/ReadVariableOp8sequential/rot_equiv_conv2d_4/convolution/ReadVariableOp2x
:sequential/rot_equiv_conv2d_4/convolution_1/ReadVariableOp:sequential/rot_equiv_conv2d_4/convolution_1/ReadVariableOp2x
:sequential/rot_equiv_conv2d_4/convolution_2/ReadVariableOp:sequential/rot_equiv_conv2d_4/convolution_2/ReadVariableOp2x
:sequential/rot_equiv_conv2d_4/convolution_3/ReadVariableOp:sequential/rot_equiv_conv2d_4/convolution_3/ReadVariableOp:i e
1
_output_shapes
:€€€€€€€€€РР
0
_user_specified_namerot_equiv_conv2d_input
С
®
3__inference_rot_equiv_conv2d_1_layer_call_fn_627363

inputs!
unknown:  
	unknown_0: 
identityИҐStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€EE *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_rot_equiv_conv2d_1_layer_call_and_return_conditional_losses_625073{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€EE `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€GG : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:€€€€€€€€€GG 
 
_user_specified_nameinputs
У
g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_627949

inputs
identityҐ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
д;
„
F__inference_sequential_layer_call_and_return_conditional_losses_625763

inputs1
rot_equiv_conv2d_625721: %
rot_equiv_conv2d_625723: 3
rot_equiv_conv2d_1_625727:  '
rot_equiv_conv2d_1_625729: 3
rot_equiv_conv2d_2_625733: @'
rot_equiv_conv2d_2_625735:@3
rot_equiv_conv2d_3_625739:@@'
rot_equiv_conv2d_3_625741:@4
rot_equiv_conv2d_4_625745:@А(
rot_equiv_conv2d_4_625747:	А
dense_625752:	А 
dense_625754:  
dense_1_625757: 
dense_1_625759:
identityИҐdense/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐ(rot_equiv_conv2d/StatefulPartitionedCallҐ*rot_equiv_conv2d_1/StatefulPartitionedCallҐ*rot_equiv_conv2d_2/StatefulPartitionedCallҐ*rot_equiv_conv2d_3/StatefulPartitionedCallҐ*rot_equiv_conv2d_4/StatefulPartitionedCall°
(rot_equiv_conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsrot_equiv_conv2d_625721rot_equiv_conv2d_625723*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:€€€€€€€€€ОО *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_rot_equiv_conv2d_layer_call_and_return_conditional_losses_624934В
 rot_equiv_pool2d/PartitionedCallPartitionedCall1rot_equiv_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€GG * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_rot_equiv_pool2d_layer_call_and_return_conditional_losses_625001 
*rot_equiv_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall)rot_equiv_pool2d/PartitionedCall:output:0rot_equiv_conv2d_1_625727rot_equiv_conv2d_1_625729*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€EE *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_rot_equiv_conv2d_1_layer_call_and_return_conditional_losses_625073И
"rot_equiv_pool2d_1/PartitionedCallPartitionedCall3rot_equiv_conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€"" * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_rot_equiv_pool2d_1_layer_call_and_return_conditional_losses_625140ћ
*rot_equiv_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall+rot_equiv_pool2d_1/PartitionedCall:output:0rot_equiv_conv2d_2_625733rot_equiv_conv2d_2_625735*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_rot_equiv_conv2d_2_layer_call_and_return_conditional_losses_625212И
"rot_equiv_pool2d_2/PartitionedCallPartitionedCall3rot_equiv_conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_rot_equiv_pool2d_2_layer_call_and_return_conditional_losses_625279ћ
*rot_equiv_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall+rot_equiv_pool2d_2/PartitionedCall:output:0rot_equiv_conv2d_3_625739rot_equiv_conv2d_3_625741*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_rot_equiv_conv2d_3_layer_call_and_return_conditional_losses_625351И
"rot_equiv_pool2d_3/PartitionedCallPartitionedCall3rot_equiv_conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_rot_equiv_pool2d_3_layer_call_and_return_conditional_losses_625418Ќ
*rot_equiv_conv2d_4/StatefulPartitionedCallStatefulPartitionedCall+rot_equiv_pool2d_3/PartitionedCall:output:0rot_equiv_conv2d_4_625745rot_equiv_conv2d_4_625747*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_rot_equiv_conv2d_4_layer_call_and_return_conditional_losses_625490щ
rot_inv_pool/PartitionedCallPartitionedCall3rot_equiv_conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_rot_inv_pool_layer_call_and_return_conditional_losses_625502ў
flatten/PartitionedCallPartitionedCall%rot_inv_pool/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_625510Б
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_625752dense_625754*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_625523П
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_625757dense_1_625759*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_625539w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€з
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall)^rot_equiv_conv2d/StatefulPartitionedCall+^rot_equiv_conv2d_1/StatefulPartitionedCall+^rot_equiv_conv2d_2/StatefulPartitionedCall+^rot_equiv_conv2d_3/StatefulPartitionedCall+^rot_equiv_conv2d_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:€€€€€€€€€РР: : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2T
(rot_equiv_conv2d/StatefulPartitionedCall(rot_equiv_conv2d/StatefulPartitionedCall2X
*rot_equiv_conv2d_1/StatefulPartitionedCall*rot_equiv_conv2d_1/StatefulPartitionedCall2X
*rot_equiv_conv2d_2/StatefulPartitionedCall*rot_equiv_conv2d_2/StatefulPartitionedCall2X
*rot_equiv_conv2d_3/StatefulPartitionedCall*rot_equiv_conv2d_3/StatefulPartitionedCall2X
*rot_equiv_conv2d_4/StatefulPartitionedCall*rot_equiv_conv2d_4/StatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€РР
 
_user_specified_nameinputs
∆6
j
N__inference_rot_equiv_pool2d_2_layer_call_and_return_conditional_losses_625279

inputs
identityG
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R K
ShapeShapeinputs*
T0*
_output_shapes
:*
out_type0	f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskG
sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RS
subSubstrided_slice:output:0sub/y:output:0*
T0	*
_output_shapes
: Z
clip_by_value/MinimumMinimumConst:output:0sub:z:0*
T0	*
_output_shapes
: Q
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R n
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*
_output_shapes
: X
GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€§
GatherV2GatherV2inputsclip_by_value:z:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€  @Ґ
max_pooling2d_2/MaxPoolMaxPoolGatherV2:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
I
Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_1Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_1Substrided_slice_1:output:0sub_1/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_1/MinimumMinimumConst_1:output:0	sub_1:z:0*
T0	*
_output_shapes
: S
clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€™

GatherV2_1GatherV2inputsclip_by_value_1:z:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€  @¶
max_pooling2d_2/MaxPool_1MaxPoolGatherV2_1:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_2Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_2StridedSliceShape_2:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_2Substrided_slice_2:output:0sub_2/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_2/MinimumMinimumConst_2:output:0	sub_2:z:0*
T0	*
_output_shapes
: S
clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€™

GatherV2_2GatherV2inputsclip_by_value_2:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€  @¶
max_pooling2d_2/MaxPool_2MaxPoolGatherV2_2:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_3Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€j
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_3StridedSliceShape_3:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_3Substrided_slice_3:output:0sub_3/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_3/MinimumMinimumConst_3:output:0	sub_3:z:0*
T0	*
_output_shapes
: S
clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_3Maximumclip_by_value_3/Minimum:z:0clip_by_value_3/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€™

GatherV2_3GatherV2inputsclip_by_value_3:z:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€  @¶
max_pooling2d_2/MaxPool_3MaxPoolGatherV2_3:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
ч
stackPack max_pooling2d_2/MaxPool:output:0"max_pooling2d_2/MaxPool_1:output:0"max_pooling2d_2/MaxPool_2:output:0"max_pooling2d_2/MaxPool_3:output:0*
N*
T0*3
_output_shapes!
:€€€€€€€€€@*
axisю€€€€€€€€b
IdentityIdentitystack:output:0*
T0*3
_output_shapes!
:€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€  @:[ W
3
_output_shapes!
:€€€€€€€€€  @
 
_user_specified_nameinputs
≤
D
(__inference_flatten_layer_call_fn_627884

inputs
identity≤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_625510a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
ј6
h
L__inference_rot_equiv_pool2d_layer_call_and_return_conditional_losses_627354

inputs
identityG
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R K
ShapeShapeinputs*
T0*
_output_shapes
:*
out_type0	f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskG
sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RS
subSubstrided_slice:output:0sub/y:output:0*
T0	*
_output_shapes
: Z
clip_by_value/MinimumMinimumConst:output:0sub:z:0*
T0	*
_output_shapes
: Q
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R n
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*
_output_shapes
: X
GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€¶
GatherV2GatherV2inputsclip_by_value:z:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*1
_output_shapes
:€€€€€€€€€ОО †
max_pooling2d/MaxPoolMaxPoolGatherV2:output:0*/
_output_shapes
:€€€€€€€€€GG *
ksize
*
paddingVALID*
strides
I
Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_1Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_1Substrided_slice_1:output:0sub_1/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_1/MinimumMinimumConst_1:output:0	sub_1:z:0*
T0	*
_output_shapes
: S
clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ђ

GatherV2_1GatherV2inputsclip_by_value_1:z:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*1
_output_shapes
:€€€€€€€€€ОО §
max_pooling2d/MaxPool_1MaxPoolGatherV2_1:output:0*/
_output_shapes
:€€€€€€€€€GG *
ksize
*
paddingVALID*
strides
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_2Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_2StridedSliceShape_2:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_2Substrided_slice_2:output:0sub_2/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_2/MinimumMinimumConst_2:output:0	sub_2:z:0*
T0	*
_output_shapes
: S
clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ђ

GatherV2_2GatherV2inputsclip_by_value_2:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*1
_output_shapes
:€€€€€€€€€ОО §
max_pooling2d/MaxPool_2MaxPoolGatherV2_2:output:0*/
_output_shapes
:€€€€€€€€€GG *
ksize
*
paddingVALID*
strides
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_3Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€j
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_3StridedSliceShape_3:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_3Substrided_slice_3:output:0sub_3/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_3/MinimumMinimumConst_3:output:0	sub_3:z:0*
T0	*
_output_shapes
: S
clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_3Maximumclip_by_value_3/Minimum:z:0clip_by_value_3/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ђ

GatherV2_3GatherV2inputsclip_by_value_3:z:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*1
_output_shapes
:€€€€€€€€€ОО §
max_pooling2d/MaxPool_3MaxPoolGatherV2_3:output:0*/
_output_shapes
:€€€€€€€€€GG *
ksize
*
paddingVALID*
strides
п
stackPackmax_pooling2d/MaxPool:output:0 max_pooling2d/MaxPool_1:output:0 max_pooling2d/MaxPool_2:output:0 max_pooling2d/MaxPool_3:output:0*
N*
T0*3
_output_shapes!
:€€€€€€€€€GG *
axisю€€€€€€€€b
IdentityIdentitystack:output:0*
T0*3
_output_shapes!
:€€€€€€€€€GG "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€ОО :] Y
5
_output_shapes#
!:€€€€€€€€€ОО 
 
_user_specified_nameinputs
∆6
j
N__inference_rot_equiv_pool2d_3_layer_call_and_return_conditional_losses_625418

inputs
identityG
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R K
ShapeShapeinputs*
T0*
_output_shapes
:*
out_type0	f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskG
sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RS
subSubstrided_slice:output:0sub/y:output:0*
T0	*
_output_shapes
: Z
clip_by_value/MinimumMinimumConst:output:0sub:z:0*
T0	*
_output_shapes
: Q
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R n
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*
_output_shapes
: X
GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€§
GatherV2GatherV2inputsclip_by_value:z:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@Ґ
max_pooling2d_3/MaxPoolMaxPoolGatherV2:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
I
Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_1Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_1Substrided_slice_1:output:0sub_1/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_1/MinimumMinimumConst_1:output:0	sub_1:z:0*
T0	*
_output_shapes
: S
clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€™

GatherV2_1GatherV2inputsclip_by_value_1:z:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@¶
max_pooling2d_3/MaxPool_1MaxPoolGatherV2_1:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_2Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_2StridedSliceShape_2:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_2Substrided_slice_2:output:0sub_2/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_2/MinimumMinimumConst_2:output:0	sub_2:z:0*
T0	*
_output_shapes
: S
clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€™

GatherV2_2GatherV2inputsclip_by_value_2:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@¶
max_pooling2d_3/MaxPool_2MaxPoolGatherV2_2:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_3Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€j
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_3StridedSliceShape_3:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_3Substrided_slice_3:output:0sub_3/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_3/MinimumMinimumConst_3:output:0	sub_3:z:0*
T0	*
_output_shapes
: S
clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_3Maximumclip_by_value_3/Minimum:z:0clip_by_value_3/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€™

GatherV2_3GatherV2inputsclip_by_value_3:z:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@¶
max_pooling2d_3/MaxPool_3MaxPoolGatherV2_3:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
ч
stackPack max_pooling2d_3/MaxPool:output:0"max_pooling2d_3/MaxPool_1:output:0"max_pooling2d_3/MaxPool_2:output:0"max_pooling2d_3/MaxPool_3:output:0*
N*
T0*3
_output_shapes!
:€€€€€€€€€@*
axisю€€€€€€€€b
IdentityIdentitystack:output:0*
T0*3
_output_shapes!
:€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€@:[ W
3
_output_shapes!
:€€€€€€€€€@
 
_user_specified_nameinputs
ј6
h
L__inference_rot_equiv_pool2d_layer_call_and_return_conditional_losses_625001

inputs
identityG
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R K
ShapeShapeinputs*
T0*
_output_shapes
:*
out_type0	f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskG
sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RS
subSubstrided_slice:output:0sub/y:output:0*
T0	*
_output_shapes
: Z
clip_by_value/MinimumMinimumConst:output:0sub:z:0*
T0	*
_output_shapes
: Q
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R n
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*
_output_shapes
: X
GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€¶
GatherV2GatherV2inputsclip_by_value:z:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*1
_output_shapes
:€€€€€€€€€ОО †
max_pooling2d/MaxPoolMaxPoolGatherV2:output:0*/
_output_shapes
:€€€€€€€€€GG *
ksize
*
paddingVALID*
strides
I
Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_1Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_1Substrided_slice_1:output:0sub_1/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_1/MinimumMinimumConst_1:output:0	sub_1:z:0*
T0	*
_output_shapes
: S
clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ђ

GatherV2_1GatherV2inputsclip_by_value_1:z:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*1
_output_shapes
:€€€€€€€€€ОО §
max_pooling2d/MaxPool_1MaxPoolGatherV2_1:output:0*/
_output_shapes
:€€€€€€€€€GG *
ksize
*
paddingVALID*
strides
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_2Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_2StridedSliceShape_2:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_2Substrided_slice_2:output:0sub_2/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_2/MinimumMinimumConst_2:output:0	sub_2:z:0*
T0	*
_output_shapes
: S
clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ђ

GatherV2_2GatherV2inputsclip_by_value_2:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*1
_output_shapes
:€€€€€€€€€ОО §
max_pooling2d/MaxPool_2MaxPoolGatherV2_2:output:0*/
_output_shapes
:€€€€€€€€€GG *
ksize
*
paddingVALID*
strides
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_3Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€j
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_3StridedSliceShape_3:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_3Substrided_slice_3:output:0sub_3/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_3/MinimumMinimumConst_3:output:0	sub_3:z:0*
T0	*
_output_shapes
: S
clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_3Maximumclip_by_value_3/Minimum:z:0clip_by_value_3/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ђ

GatherV2_3GatherV2inputsclip_by_value_3:z:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*1
_output_shapes
:€€€€€€€€€ОО §
max_pooling2d/MaxPool_3MaxPoolGatherV2_3:output:0*/
_output_shapes
:€€€€€€€€€GG *
ksize
*
paddingVALID*
strides
п
stackPackmax_pooling2d/MaxPool:output:0 max_pooling2d/MaxPool_1:output:0 max_pooling2d/MaxPool_2:output:0 max_pooling2d/MaxPool_3:output:0*
N*
T0*3
_output_shapes!
:€€€€€€€€€GG *
axisю€€€€€€€€b
IdentityIdentitystack:output:0*
T0*3
_output_shapes!
:€€€€€€€€€GG "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€ОО :] Y
5
_output_shapes#
!:€€€€€€€€€ОО 
 
_user_specified_nameinputs
У
g
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_624833

inputs
identityҐ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
еC
о
N__inference_rot_equiv_conv2d_3_layer_call_and_return_conditional_losses_625351

inputs=
#convolution_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐconvolution/ReadVariableOpҐconvolution_1/ReadVariableOpҐconvolution_2/ReadVariableOpҐconvolution_3/ReadVariableOpG
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R K
ShapeShapeinputs*
T0*
_output_shapes
:*
out_type0	f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskG
sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RS
subSubstrided_slice:output:0sub/y:output:0*
T0	*
_output_shapes
: Z
clip_by_value/MinimumMinimumConst:output:0sub:z:0*
T0	*
_output_shapes
: Q
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R n
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*
_output_shapes
: X
GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€§
GatherV2GatherV2inputsclip_by_value:z:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@Ж
convolution/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype0ѓ
convolutionConv2DGatherV2:output:0"convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingVALID*
strides
I
Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_1Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_1Substrided_slice_1:output:0sub_1/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_1/MinimumMinimumConst_1:output:0	sub_1:z:0*
T0	*
_output_shapes
: S
clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€™

GatherV2_1GatherV2inputsclip_by_value_1:z:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@И
convolution_1/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype0µ
convolution_1Conv2DGatherV2_1:output:0$convolution_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingVALID*
strides
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_2Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_2StridedSliceShape_2:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_2Substrided_slice_2:output:0sub_2/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_2/MinimumMinimumConst_2:output:0	sub_2:z:0*
T0	*
_output_shapes
: S
clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€™

GatherV2_2GatherV2inputsclip_by_value_2:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@И
convolution_2/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype0µ
convolution_2Conv2DGatherV2_2:output:0$convolution_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingVALID*
strides
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_3Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€j
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_3StridedSliceShape_3:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_3Substrided_slice_3:output:0sub_3/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_3/MinimumMinimumConst_3:output:0	sub_3:z:0*
T0	*
_output_shapes
: S
clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_3Maximumclip_by_value_3/Minimum:z:0clip_by_value_3/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€™

GatherV2_3GatherV2inputsclip_by_value_3:z:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@И
convolution_3/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype0µ
convolution_3Conv2DGatherV2_3:output:0$convolution_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingVALID*
strides
«
stackPackconvolution:output:0convolution_1:output:0convolution_2:output:0convolution_3:output:0*
N*
T0*3
_output_shapes!
:€€€€€€€€€@*
axisю€€€€€€€€Z
ReluRelustack:output:0*
T0*3
_output_shapes!
:€€€€€€€€€@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Д
BiasAddBiasAddRelu:activations:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€@k
IdentityIdentityBiasAdd:output:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€@ў
NoOpNoOp^BiasAdd/ReadVariableOp^convolution/ReadVariableOp^convolution_1/ReadVariableOp^convolution_2/ReadVariableOp^convolution_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp28
convolution/ReadVariableOpconvolution/ReadVariableOp2<
convolution_1/ReadVariableOpconvolution_1/ReadVariableOp2<
convolution_2/ReadVariableOpconvolution_2/ReadVariableOp2<
convolution_3/ReadVariableOpconvolution_3/ReadVariableOp:[ W
3
_output_shapes!
:€€€€€€€€€@
 
_user_specified_nameinputs
еC
о
N__inference_rot_equiv_conv2d_1_layer_call_and_return_conditional_losses_625073

inputs=
#convolution_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐconvolution/ReadVariableOpҐconvolution_1/ReadVariableOpҐconvolution_2/ReadVariableOpҐconvolution_3/ReadVariableOpG
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R K
ShapeShapeinputs*
T0*
_output_shapes
:*
out_type0	f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskG
sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RS
subSubstrided_slice:output:0sub/y:output:0*
T0	*
_output_shapes
: Z
clip_by_value/MinimumMinimumConst:output:0sub:z:0*
T0	*
_output_shapes
: Q
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R n
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*
_output_shapes
: X
GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€§
GatherV2GatherV2inputsclip_by_value:z:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€GG Ж
convolution/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:  *
dtype0ѓ
convolutionConv2DGatherV2:output:0"convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€EE *
paddingVALID*
strides
I
Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_1Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_1Substrided_slice_1:output:0sub_1/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_1/MinimumMinimumConst_1:output:0	sub_1:z:0*
T0	*
_output_shapes
: S
clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€™

GatherV2_1GatherV2inputsclip_by_value_1:z:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€GG И
convolution_1/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:  *
dtype0µ
convolution_1Conv2DGatherV2_1:output:0$convolution_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€EE *
paddingVALID*
strides
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_2Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_2StridedSliceShape_2:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_2Substrided_slice_2:output:0sub_2/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_2/MinimumMinimumConst_2:output:0	sub_2:z:0*
T0	*
_output_shapes
: S
clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€™

GatherV2_2GatherV2inputsclip_by_value_2:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€GG И
convolution_2/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:  *
dtype0µ
convolution_2Conv2DGatherV2_2:output:0$convolution_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€EE *
paddingVALID*
strides
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_3Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€j
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_3StridedSliceShape_3:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_3Substrided_slice_3:output:0sub_3/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_3/MinimumMinimumConst_3:output:0	sub_3:z:0*
T0	*
_output_shapes
: S
clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_3Maximumclip_by_value_3/Minimum:z:0clip_by_value_3/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€™

GatherV2_3GatherV2inputsclip_by_value_3:z:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€GG И
convolution_3/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:  *
dtype0µ
convolution_3Conv2DGatherV2_3:output:0$convolution_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€EE *
paddingVALID*
strides
«
stackPackconvolution:output:0convolution_1:output:0convolution_2:output:0convolution_3:output:0*
N*
T0*3
_output_shapes!
:€€€€€€€€€EE *
axisю€€€€€€€€Z
ReluRelustack:output:0*
T0*3
_output_shapes!
:€€€€€€€€€EE r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0Д
BiasAddBiasAddRelu:activations:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€EE k
IdentityIdentityBiasAdd:output:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€EE ў
NoOpNoOp^BiasAdd/ReadVariableOp^convolution/ReadVariableOp^convolution_1/ReadVariableOp^convolution_2/ReadVariableOp^convolution_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€GG : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp28
convolution/ReadVariableOpconvolution/ReadVariableOp2<
convolution_1/ReadVariableOpconvolution_1/ReadVariableOp2<
convolution_2/ReadVariableOpconvolution_2/ReadVariableOp2<
convolution_3/ReadVariableOpconvolution_3/ReadVariableOp:[ W
3
_output_shapes!
:€€€€€€€€€GG 
 
_user_specified_nameinputs
Ж
К
+__inference_sequential_layer_call_fn_626024

inputs!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@$
	unknown_7:@А
	unknown_8:	А
	unknown_9:	А 

unknown_10: 

unknown_11: 

unknown_12:
identityИҐStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_625763o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:€€€€€€€€€РР: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€РР
 
_user_specified_nameinputs
д
O
3__inference_rot_equiv_pool2d_3_layer_call_fn_627728

inputs
identity»
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_rot_equiv_pool2d_3_layer_call_and_return_conditional_losses_625418l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€@:[ W
3
_output_shapes!
:€€€€€€€€€@
 
_user_specified_nameinputs
∆6
j
N__inference_rot_equiv_pool2d_1_layer_call_and_return_conditional_losses_625140

inputs
identityG
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R K
ShapeShapeinputs*
T0*
_output_shapes
:*
out_type0	f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskG
sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RS
subSubstrided_slice:output:0sub/y:output:0*
T0	*
_output_shapes
: Z
clip_by_value/MinimumMinimumConst:output:0sub:z:0*
T0	*
_output_shapes
: Q
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R n
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*
_output_shapes
: X
GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€§
GatherV2GatherV2inputsclip_by_value:z:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€EE Ґ
max_pooling2d_1/MaxPoolMaxPoolGatherV2:output:0*/
_output_shapes
:€€€€€€€€€"" *
ksize
*
paddingVALID*
strides
I
Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_1Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_1Substrided_slice_1:output:0sub_1/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_1/MinimumMinimumConst_1:output:0	sub_1:z:0*
T0	*
_output_shapes
: S
clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€™

GatherV2_1GatherV2inputsclip_by_value_1:z:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€EE ¶
max_pooling2d_1/MaxPool_1MaxPoolGatherV2_1:output:0*/
_output_shapes
:€€€€€€€€€"" *
ksize
*
paddingVALID*
strides
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_2Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_2StridedSliceShape_2:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_2Substrided_slice_2:output:0sub_2/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_2/MinimumMinimumConst_2:output:0	sub_2:z:0*
T0	*
_output_shapes
: S
clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€™

GatherV2_2GatherV2inputsclip_by_value_2:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€EE ¶
max_pooling2d_1/MaxPool_2MaxPoolGatherV2_2:output:0*/
_output_shapes
:€€€€€€€€€"" *
ksize
*
paddingVALID*
strides
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_3Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€j
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_3StridedSliceShape_3:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_3Substrided_slice_3:output:0sub_3/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_3/MinimumMinimumConst_3:output:0	sub_3:z:0*
T0	*
_output_shapes
: S
clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_3Maximumclip_by_value_3/Minimum:z:0clip_by_value_3/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€™

GatherV2_3GatherV2inputsclip_by_value_3:z:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€EE ¶
max_pooling2d_1/MaxPool_3MaxPoolGatherV2_3:output:0*/
_output_shapes
:€€€€€€€€€"" *
ksize
*
paddingVALID*
strides
ч
stackPack max_pooling2d_1/MaxPool:output:0"max_pooling2d_1/MaxPool_1:output:0"max_pooling2d_1/MaxPool_2:output:0"max_pooling2d_1/MaxPool_3:output:0*
N*
T0*3
_output_shapes!
:€€€€€€€€€"" *
axisю€€€€€€€€b
IdentityIdentitystack:output:0*
T0*3
_output_shapes!
:€€€€€€€€€"" "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€EE :[ W
3
_output_shapes!
:€€€€€€€€€EE 
 
_user_specified_nameinputs
фC
р
N__inference_rot_equiv_conv2d_4_layer_call_and_return_conditional_losses_627868

inputs>
#convolution_readvariableop_resource:@А.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐconvolution/ReadVariableOpҐconvolution_1/ReadVariableOpҐconvolution_2/ReadVariableOpҐconvolution_3/ReadVariableOpG
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R K
ShapeShapeinputs*
T0*
_output_shapes
:*
out_type0	f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskG
sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RS
subSubstrided_slice:output:0sub/y:output:0*
T0	*
_output_shapes
: Z
clip_by_value/MinimumMinimumConst:output:0sub:z:0*
T0	*
_output_shapes
: Q
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R n
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*
_output_shapes
: X
GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€§
GatherV2GatherV2inputsclip_by_value:z:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@З
convolution/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*'
_output_shapes
:@А*
dtype0∞
convolutionConv2DGatherV2:output:0"convolution/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingVALID*
strides
I
Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_1Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_1Substrided_slice_1:output:0sub_1/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_1/MinimumMinimumConst_1:output:0	sub_1:z:0*
T0	*
_output_shapes
: S
clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€™

GatherV2_1GatherV2inputsclip_by_value_1:z:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@Й
convolution_1/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*'
_output_shapes
:@А*
dtype0ґ
convolution_1Conv2DGatherV2_1:output:0$convolution_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingVALID*
strides
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_2Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_2StridedSliceShape_2:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_2Substrided_slice_2:output:0sub_2/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_2/MinimumMinimumConst_2:output:0	sub_2:z:0*
T0	*
_output_shapes
: S
clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€™

GatherV2_2GatherV2inputsclip_by_value_2:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@Й
convolution_2/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*'
_output_shapes
:@А*
dtype0ґ
convolution_2Conv2DGatherV2_2:output:0$convolution_2/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingVALID*
strides
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_3Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€j
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_3StridedSliceShape_3:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_3Substrided_slice_3:output:0sub_3/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_3/MinimumMinimumConst_3:output:0	sub_3:z:0*
T0	*
_output_shapes
: S
clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_3Maximumclip_by_value_3/Minimum:z:0clip_by_value_3/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€™

GatherV2_3GatherV2inputsclip_by_value_3:z:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@Й
convolution_3/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*'
_output_shapes
:@А*
dtype0ґ
convolution_3Conv2DGatherV2_3:output:0$convolution_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingVALID*
strides
»
stackPackconvolution:output:0convolution_1:output:0convolution_2:output:0convolution_3:output:0*
N*
T0*4
_output_shapes"
 :€€€€€€€€€А*
axisю€€€€€€€€[
ReluRelustack:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Е
BiasAddBiasAddRelu:activations:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аl
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€Аў
NoOpNoOp^BiasAdd/ReadVariableOp^convolution/ReadVariableOp^convolution_1/ReadVariableOp^convolution_2/ReadVariableOp^convolution_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp28
convolution/ReadVariableOpconvolution/ReadVariableOp2<
convolution_1/ReadVariableOpconvolution_1/ReadVariableOp2<
convolution_2/ReadVariableOpconvolution_2/ReadVariableOp2<
convolution_3/ReadVariableOpconvolution_3/ReadVariableOp:[ W
3
_output_shapes!
:€€€€€€€€€@
 
_user_specified_nameinputs
д
O
3__inference_rot_equiv_pool2d_1_layer_call_fn_627438

inputs
identity»
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€"" * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_rot_equiv_pool2d_1_layer_call_and_return_conditional_losses_625140l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:€€€€€€€€€"" "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€EE :[ W
3
_output_shapes!
:€€€€€€€€€EE 
 
_user_specified_nameinputs
Ь

у
A__inference_dense_layer_call_and_return_conditional_losses_625523

inputs1
matmul_readvariableop_resource:	А -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
еC
о
N__inference_rot_equiv_conv2d_3_layer_call_and_return_conditional_losses_627723

inputs=
#convolution_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐconvolution/ReadVariableOpҐconvolution_1/ReadVariableOpҐconvolution_2/ReadVariableOpҐconvolution_3/ReadVariableOpG
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R K
ShapeShapeinputs*
T0*
_output_shapes
:*
out_type0	f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskG
sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RS
subSubstrided_slice:output:0sub/y:output:0*
T0	*
_output_shapes
: Z
clip_by_value/MinimumMinimumConst:output:0sub:z:0*
T0	*
_output_shapes
: Q
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R n
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*
_output_shapes
: X
GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€§
GatherV2GatherV2inputsclip_by_value:z:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@Ж
convolution/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype0ѓ
convolutionConv2DGatherV2:output:0"convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingVALID*
strides
I
Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_1Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_1Substrided_slice_1:output:0sub_1/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_1/MinimumMinimumConst_1:output:0	sub_1:z:0*
T0	*
_output_shapes
: S
clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€™

GatherV2_1GatherV2inputsclip_by_value_1:z:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@И
convolution_1/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype0µ
convolution_1Conv2DGatherV2_1:output:0$convolution_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingVALID*
strides
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_2Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_2StridedSliceShape_2:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_2Substrided_slice_2:output:0sub_2/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_2/MinimumMinimumConst_2:output:0	sub_2:z:0*
T0	*
_output_shapes
: S
clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€™

GatherV2_2GatherV2inputsclip_by_value_2:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@И
convolution_2/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype0µ
convolution_2Conv2DGatherV2_2:output:0$convolution_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingVALID*
strides
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_3Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€j
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_3StridedSliceShape_3:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_3Substrided_slice_3:output:0sub_3/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_3/MinimumMinimumConst_3:output:0	sub_3:z:0*
T0	*
_output_shapes
: S
clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_3Maximumclip_by_value_3/Minimum:z:0clip_by_value_3/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€™

GatherV2_3GatherV2inputsclip_by_value_3:z:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@И
convolution_3/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype0µ
convolution_3Conv2DGatherV2_3:output:0$convolution_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingVALID*
strides
«
stackPackconvolution:output:0convolution_1:output:0convolution_2:output:0convolution_3:output:0*
N*
T0*3
_output_shapes!
:€€€€€€€€€@*
axisю€€€€€€€€Z
ReluRelustack:output:0*
T0*3
_output_shapes!
:€€€€€€€€€@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Д
BiasAddBiasAddRelu:activations:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€@k
IdentityIdentityBiasAdd:output:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€@ў
NoOpNoOp^BiasAdd/ReadVariableOp^convolution/ReadVariableOp^convolution_1/ReadVariableOp^convolution_2/ReadVariableOp^convolution_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp28
convolution/ReadVariableOpconvolution/ReadVariableOp2<
convolution_1/ReadVariableOpconvolution_1/ReadVariableOp2<
convolution_2/ReadVariableOpconvolution_2/ReadVariableOp2<
convolution_3/ReadVariableOpconvolution_3/ReadVariableOp:[ W
3
_output_shapes!
:€€€€€€€€€@
 
_user_specified_nameinputs
≥и
Ѓ
F__inference_sequential_layer_call_and_return_conditional_losses_626612

inputsN
4rot_equiv_conv2d_convolution_readvariableop_resource: >
0rot_equiv_conv2d_biasadd_readvariableop_resource: P
6rot_equiv_conv2d_1_convolution_readvariableop_resource:  @
2rot_equiv_conv2d_1_biasadd_readvariableop_resource: P
6rot_equiv_conv2d_2_convolution_readvariableop_resource: @@
2rot_equiv_conv2d_2_biasadd_readvariableop_resource:@P
6rot_equiv_conv2d_3_convolution_readvariableop_resource:@@@
2rot_equiv_conv2d_3_biasadd_readvariableop_resource:@Q
6rot_equiv_conv2d_4_convolution_readvariableop_resource:@АA
2rot_equiv_conv2d_4_biasadd_readvariableop_resource:	А7
$dense_matmul_readvariableop_resource:	А 3
%dense_biasadd_readvariableop_resource: 8
&dense_1_matmul_readvariableop_resource: 5
'dense_1_biasadd_readvariableop_resource:
identityИҐdense/BiasAdd/ReadVariableOpҐdense/MatMul/ReadVariableOpҐdense_1/BiasAdd/ReadVariableOpҐdense_1/MatMul/ReadVariableOpҐ'rot_equiv_conv2d/BiasAdd/ReadVariableOpҐrot_equiv_conv2d/ReadVariableOpҐ!rot_equiv_conv2d/ReadVariableOp_1Ґ!rot_equiv_conv2d/ReadVariableOp_2Ґ+rot_equiv_conv2d/convolution/ReadVariableOpҐ)rot_equiv_conv2d_1/BiasAdd/ReadVariableOpҐ-rot_equiv_conv2d_1/convolution/ReadVariableOpҐ/rot_equiv_conv2d_1/convolution_1/ReadVariableOpҐ/rot_equiv_conv2d_1/convolution_2/ReadVariableOpҐ/rot_equiv_conv2d_1/convolution_3/ReadVariableOpҐ)rot_equiv_conv2d_2/BiasAdd/ReadVariableOpҐ-rot_equiv_conv2d_2/convolution/ReadVariableOpҐ/rot_equiv_conv2d_2/convolution_1/ReadVariableOpҐ/rot_equiv_conv2d_2/convolution_2/ReadVariableOpҐ/rot_equiv_conv2d_2/convolution_3/ReadVariableOpҐ)rot_equiv_conv2d_3/BiasAdd/ReadVariableOpҐ-rot_equiv_conv2d_3/convolution/ReadVariableOpҐ/rot_equiv_conv2d_3/convolution_1/ReadVariableOpҐ/rot_equiv_conv2d_3/convolution_2/ReadVariableOpҐ/rot_equiv_conv2d_3/convolution_3/ReadVariableOpҐ)rot_equiv_conv2d_4/BiasAdd/ReadVariableOpҐ-rot_equiv_conv2d_4/convolution/ReadVariableOpҐ/rot_equiv_conv2d_4/convolution_1/ReadVariableOpҐ/rot_equiv_conv2d_4/convolution_2/ReadVariableOpҐ/rot_equiv_conv2d_4/convolution_3/ReadVariableOp®
+rot_equiv_conv2d/convolution/ReadVariableOpReadVariableOp4rot_equiv_conv2d_convolution_readvariableop_resource*&
_output_shapes
: *
dtype0»
rot_equiv_conv2d/convolutionConv2Dinputs3rot_equiv_conv2d/convolution/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ОО *
paddingVALID*
strides
°
$rot_equiv_conv2d/Rank/ReadVariableOpReadVariableOp4rot_equiv_conv2d_convolution_readvariableop_resource*&
_output_shapes
: *
dtype0W
rot_equiv_conv2d/RankConst*
_output_shapes
: *
dtype0*
value	B :^
rot_equiv_conv2d/range/startConst*
_output_shapes
: *
dtype0*
value	B : ^
rot_equiv_conv2d/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :©
rot_equiv_conv2d/rangeRange%rot_equiv_conv2d/range/start:output:0rot_equiv_conv2d/Rank:output:0%rot_equiv_conv2d/range/delta:output:0*
_output_shapes
:Е
,rot_equiv_conv2d/TensorScatterUpdate/indicesConst*
_output_shapes

:*
dtype0*!
valueB"       }
,rot_equiv_conv2d/TensorScatterUpdate/updatesConst*
_output_shapes
:*
dtype0*
valueB"       €
$rot_equiv_conv2d/TensorScatterUpdateTensorScatterUpdaterot_equiv_conv2d/range:output:05rot_equiv_conv2d/TensorScatterUpdate/indices:output:05rot_equiv_conv2d/TensorScatterUpdate/updates:output:0*
T0*
Tindices0*
_output_shapes
:Ь
rot_equiv_conv2d/ReadVariableOpReadVariableOp4rot_equiv_conv2d_convolution_readvariableop_resource*&
_output_shapes
: *
dtype0Y
rot_equiv_conv2d/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :i
rot_equiv_conv2d/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:Ђ
rot_equiv_conv2d/ReverseV2	ReverseV2'rot_equiv_conv2d/ReadVariableOp:value:0(rot_equiv_conv2d/ReverseV2/axis:output:0*
T0*&
_output_shapes
: ђ
rot_equiv_conv2d/transpose	Transpose#rot_equiv_conv2d/ReverseV2:output:0-rot_equiv_conv2d/TensorScatterUpdate:output:0*
T0*&
_output_shapes
: µ
rot_equiv_conv2d/convolution_1Conv2Dinputsrot_equiv_conv2d/transpose:y:0*
T0*1
_output_shapes
:€€€€€€€€€ОО *
paddingVALID*
strides
Y
rot_equiv_conv2d/Rank_2Const*
_output_shapes
: *
dtype0*
value	B :`
rot_equiv_conv2d/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : `
rot_equiv_conv2d/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :±
rot_equiv_conv2d/range_1Range'rot_equiv_conv2d/range_1/start:output:0 rot_equiv_conv2d/Rank_2:output:0'rot_equiv_conv2d/range_1/delta:output:0*
_output_shapes
:З
.rot_equiv_conv2d/TensorScatterUpdate_1/indicesConst*
_output_shapes

:*
dtype0*!
valueB"      
.rot_equiv_conv2d/TensorScatterUpdate_1/updatesConst*
_output_shapes
:*
dtype0*
valueB"      З
&rot_equiv_conv2d/TensorScatterUpdate_1TensorScatterUpdate!rot_equiv_conv2d/range_1:output:07rot_equiv_conv2d/TensorScatterUpdate_1/indices:output:07rot_equiv_conv2d/TensorScatterUpdate_1/updates:output:0*
T0*
Tindices0*
_output_shapes
:њ
rot_equiv_conv2d/transpose_1	Transpose'rot_equiv_conv2d/convolution_1:output:0/rot_equiv_conv2d/TensorScatterUpdate_1:output:0*
T0*1
_output_shapes
:€€€€€€€€€ОО Y
rot_equiv_conv2d/Rank_3Const*
_output_shapes
: *
dtype0*
value	B :k
!rot_equiv_conv2d/ReverseV2_1/axisConst*
_output_shapes
:*
dtype0*
valueB:≥
rot_equiv_conv2d/ReverseV2_1	ReverseV2 rot_equiv_conv2d/transpose_1:y:0*rot_equiv_conv2d/ReverseV2_1/axis:output:0*
T0*1
_output_shapes
:€€€€€€€€€ОО £
&rot_equiv_conv2d/Rank_4/ReadVariableOpReadVariableOp4rot_equiv_conv2d_convolution_readvariableop_resource*&
_output_shapes
: *
dtype0Y
rot_equiv_conv2d/Rank_4Const*
_output_shapes
: *
dtype0*
value	B :Ю
!rot_equiv_conv2d/ReadVariableOp_1ReadVariableOp4rot_equiv_conv2d_convolution_readvariableop_resource*&
_output_shapes
: *
dtype0Y
rot_equiv_conv2d/Rank_5Const*
_output_shapes
: *
dtype0*
value	B :k
!rot_equiv_conv2d/ReverseV2_2/axisConst*
_output_shapes
:*
dtype0*
valueB: ±
rot_equiv_conv2d/ReverseV2_2	ReverseV2)rot_equiv_conv2d/ReadVariableOp_1:value:0*rot_equiv_conv2d/ReverseV2_2/axis:output:0*
T0*&
_output_shapes
: Y
rot_equiv_conv2d/Rank_6Const*
_output_shapes
: *
dtype0*
value	B :k
!rot_equiv_conv2d/ReverseV2_3/axisConst*
_output_shapes
:*
dtype0*
valueB:≠
rot_equiv_conv2d/ReverseV2_3	ReverseV2%rot_equiv_conv2d/ReverseV2_2:output:0*rot_equiv_conv2d/ReverseV2_3/axis:output:0*
T0*&
_output_shapes
: Љ
rot_equiv_conv2d/convolution_2Conv2Dinputs%rot_equiv_conv2d/ReverseV2_3:output:0*
T0*1
_output_shapes
:€€€€€€€€€ОО *
paddingVALID*
strides
Y
rot_equiv_conv2d/Rank_7Const*
_output_shapes
: *
dtype0*
value	B :Y
rot_equiv_conv2d/Rank_8Const*
_output_shapes
: *
dtype0*
value	B :k
!rot_equiv_conv2d/ReverseV2_4/axisConst*
_output_shapes
:*
dtype0*
valueB:Ї
rot_equiv_conv2d/ReverseV2_4	ReverseV2'rot_equiv_conv2d/convolution_2:output:0*rot_equiv_conv2d/ReverseV2_4/axis:output:0*
T0*1
_output_shapes
:€€€€€€€€€ОО Y
rot_equiv_conv2d/Rank_9Const*
_output_shapes
: *
dtype0*
value	B :k
!rot_equiv_conv2d/ReverseV2_5/axisConst*
_output_shapes
:*
dtype0*
valueB:Є
rot_equiv_conv2d/ReverseV2_5	ReverseV2%rot_equiv_conv2d/ReverseV2_4:output:0*rot_equiv_conv2d/ReverseV2_5/axis:output:0*
T0*1
_output_shapes
:€€€€€€€€€ОО §
'rot_equiv_conv2d/Rank_10/ReadVariableOpReadVariableOp4rot_equiv_conv2d_convolution_readvariableop_resource*&
_output_shapes
: *
dtype0Z
rot_equiv_conv2d/Rank_10Const*
_output_shapes
: *
dtype0*
value	B :`
rot_equiv_conv2d/range_2/startConst*
_output_shapes
: *
dtype0*
value	B : `
rot_equiv_conv2d/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :≤
rot_equiv_conv2d/range_2Range'rot_equiv_conv2d/range_2/start:output:0!rot_equiv_conv2d/Rank_10:output:0'rot_equiv_conv2d/range_2/delta:output:0*
_output_shapes
:З
.rot_equiv_conv2d/TensorScatterUpdate_2/indicesConst*
_output_shapes

:*
dtype0*!
valueB"       
.rot_equiv_conv2d/TensorScatterUpdate_2/updatesConst*
_output_shapes
:*
dtype0*
valueB"       З
&rot_equiv_conv2d/TensorScatterUpdate_2TensorScatterUpdate!rot_equiv_conv2d/range_2:output:07rot_equiv_conv2d/TensorScatterUpdate_2/indices:output:07rot_equiv_conv2d/TensorScatterUpdate_2/updates:output:0*
T0*
Tindices0*
_output_shapes
:Ю
!rot_equiv_conv2d/ReadVariableOp_2ReadVariableOp4rot_equiv_conv2d_convolution_readvariableop_resource*&
_output_shapes
: *
dtype0ґ
rot_equiv_conv2d/transpose_2	Transpose)rot_equiv_conv2d/ReadVariableOp_2:value:0/rot_equiv_conv2d/TensorScatterUpdate_2:output:0*
T0*&
_output_shapes
: Z
rot_equiv_conv2d/Rank_11Const*
_output_shapes
: *
dtype0*
value	B :k
!rot_equiv_conv2d/ReverseV2_6/axisConst*
_output_shapes
:*
dtype0*
valueB:®
rot_equiv_conv2d/ReverseV2_6	ReverseV2 rot_equiv_conv2d/transpose_2:y:0*rot_equiv_conv2d/ReverseV2_6/axis:output:0*
T0*&
_output_shapes
: Љ
rot_equiv_conv2d/convolution_3Conv2Dinputs%rot_equiv_conv2d/ReverseV2_6:output:0*
T0*1
_output_shapes
:€€€€€€€€€ОО *
paddingVALID*
strides
Z
rot_equiv_conv2d/Rank_12Const*
_output_shapes
: *
dtype0*
value	B :`
rot_equiv_conv2d/range_3/startConst*
_output_shapes
: *
dtype0*
value	B : `
rot_equiv_conv2d/range_3/deltaConst*
_output_shapes
: *
dtype0*
value	B :≤
rot_equiv_conv2d/range_3Range'rot_equiv_conv2d/range_3/start:output:0!rot_equiv_conv2d/Rank_12:output:0'rot_equiv_conv2d/range_3/delta:output:0*
_output_shapes
:З
.rot_equiv_conv2d/TensorScatterUpdate_3/indicesConst*
_output_shapes

:*
dtype0*!
valueB"      
.rot_equiv_conv2d/TensorScatterUpdate_3/updatesConst*
_output_shapes
:*
dtype0*
valueB"      З
&rot_equiv_conv2d/TensorScatterUpdate_3TensorScatterUpdate!rot_equiv_conv2d/range_3:output:07rot_equiv_conv2d/TensorScatterUpdate_3/indices:output:07rot_equiv_conv2d/TensorScatterUpdate_3/updates:output:0*
T0*
Tindices0*
_output_shapes
:Z
rot_equiv_conv2d/Rank_13Const*
_output_shapes
: *
dtype0*
value	B :k
!rot_equiv_conv2d/ReverseV2_7/axisConst*
_output_shapes
:*
dtype0*
valueB:Ї
rot_equiv_conv2d/ReverseV2_7	ReverseV2'rot_equiv_conv2d/convolution_3:output:0*rot_equiv_conv2d/ReverseV2_7/axis:output:0*
T0*1
_output_shapes
:€€€€€€€€€ОО љ
rot_equiv_conv2d/transpose_3	Transpose%rot_equiv_conv2d/ReverseV2_7:output:0/rot_equiv_conv2d/TensorScatterUpdate_3:output:0*
T0*1
_output_shapes
:€€€€€€€€€ОО У
rot_equiv_conv2d/stackPack%rot_equiv_conv2d/convolution:output:0%rot_equiv_conv2d/ReverseV2_1:output:0%rot_equiv_conv2d/ReverseV2_5:output:0 rot_equiv_conv2d/transpose_3:y:0*
N*
T0*5
_output_shapes#
!:€€€€€€€€€ОО *
axisю€€€€€€€€~
rot_equiv_conv2d/ReluRelurot_equiv_conv2d/stack:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€ОО Ф
'rot_equiv_conv2d/BiasAdd/ReadVariableOpReadVariableOp0rot_equiv_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0є
rot_equiv_conv2d/BiasAddBiasAdd#rot_equiv_conv2d/Relu:activations:0/rot_equiv_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:€€€€€€€€€ОО X
rot_equiv_pool2d/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R w
rot_equiv_pool2d/ShapeShape!rot_equiv_conv2d/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	w
$rot_equiv_pool2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€y
&rot_equiv_pool2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€p
&rot_equiv_pool2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¶
rot_equiv_pool2d/strided_sliceStridedSlicerot_equiv_pool2d/Shape:output:0-rot_equiv_pool2d/strided_slice/stack:output:0/rot_equiv_pool2d/strided_slice/stack_1:output:0/rot_equiv_pool2d/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskX
rot_equiv_pool2d/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RЖ
rot_equiv_pool2d/subSub'rot_equiv_pool2d/strided_slice:output:0rot_equiv_pool2d/sub/y:output:0*
T0	*
_output_shapes
: Н
&rot_equiv_pool2d/clip_by_value/MinimumMinimumrot_equiv_pool2d/Const:output:0rot_equiv_pool2d/sub:z:0*
T0	*
_output_shapes
: b
 rot_equiv_pool2d/clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R °
rot_equiv_pool2d/clip_by_valueMaximum*rot_equiv_pool2d/clip_by_value/Minimum:z:0)rot_equiv_pool2d/clip_by_value/y:output:0*
T0	*
_output_shapes
: i
rot_equiv_pool2d/GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ф
rot_equiv_pool2d/GatherV2GatherV2!rot_equiv_conv2d/BiasAdd:output:0"rot_equiv_pool2d/clip_by_value:z:0'rot_equiv_pool2d/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*1
_output_shapes
:€€€€€€€€€ОО ¬
&rot_equiv_pool2d/max_pooling2d/MaxPoolMaxPool"rot_equiv_pool2d/GatherV2:output:0*/
_output_shapes
:€€€€€€€€€GG *
ksize
*
paddingVALID*
strides
Z
rot_equiv_pool2d/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 Ry
rot_equiv_pool2d/Shape_1Shape!rot_equiv_conv2d/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	y
&rot_equiv_pool2d/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€{
(rot_equiv_pool2d/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€r
(rot_equiv_pool2d/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:∞
 rot_equiv_pool2d/strided_slice_1StridedSlice!rot_equiv_pool2d/Shape_1:output:0/rot_equiv_pool2d/strided_slice_1/stack:output:01rot_equiv_pool2d/strided_slice_1/stack_1:output:01rot_equiv_pool2d/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskZ
rot_equiv_pool2d/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 RМ
rot_equiv_pool2d/sub_1Sub)rot_equiv_pool2d/strided_slice_1:output:0!rot_equiv_pool2d/sub_1/y:output:0*
T0	*
_output_shapes
: У
(rot_equiv_pool2d/clip_by_value_1/MinimumMinimum!rot_equiv_pool2d/Const_1:output:0rot_equiv_pool2d/sub_1:z:0*
T0	*
_output_shapes
: d
"rot_equiv_pool2d/clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R І
 rot_equiv_pool2d/clip_by_value_1Maximum,rot_equiv_pool2d/clip_by_value_1/Minimum:z:0+rot_equiv_pool2d/clip_by_value_1/y:output:0*
T0	*
_output_shapes
: k
 rot_equiv_pool2d/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ъ
rot_equiv_pool2d/GatherV2_1GatherV2!rot_equiv_conv2d/BiasAdd:output:0$rot_equiv_pool2d/clip_by_value_1:z:0)rot_equiv_pool2d/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*1
_output_shapes
:€€€€€€€€€ОО ∆
(rot_equiv_pool2d/max_pooling2d/MaxPool_1MaxPool$rot_equiv_pool2d/GatherV2_1:output:0*/
_output_shapes
:€€€€€€€€€GG *
ksize
*
paddingVALID*
strides
Z
rot_equiv_pool2d/Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 Ry
rot_equiv_pool2d/Shape_2Shape!rot_equiv_conv2d/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	y
&rot_equiv_pool2d/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€{
(rot_equiv_pool2d/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€r
(rot_equiv_pool2d/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:∞
 rot_equiv_pool2d/strided_slice_2StridedSlice!rot_equiv_pool2d/Shape_2:output:0/rot_equiv_pool2d/strided_slice_2/stack:output:01rot_equiv_pool2d/strided_slice_2/stack_1:output:01rot_equiv_pool2d/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskZ
rot_equiv_pool2d/sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 RМ
rot_equiv_pool2d/sub_2Sub)rot_equiv_pool2d/strided_slice_2:output:0!rot_equiv_pool2d/sub_2/y:output:0*
T0	*
_output_shapes
: У
(rot_equiv_pool2d/clip_by_value_2/MinimumMinimum!rot_equiv_pool2d/Const_2:output:0rot_equiv_pool2d/sub_2:z:0*
T0	*
_output_shapes
: d
"rot_equiv_pool2d/clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R І
 rot_equiv_pool2d/clip_by_value_2Maximum,rot_equiv_pool2d/clip_by_value_2/Minimum:z:0+rot_equiv_pool2d/clip_by_value_2/y:output:0*
T0	*
_output_shapes
: k
 rot_equiv_pool2d/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ъ
rot_equiv_pool2d/GatherV2_2GatherV2!rot_equiv_conv2d/BiasAdd:output:0$rot_equiv_pool2d/clip_by_value_2:z:0)rot_equiv_pool2d/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*1
_output_shapes
:€€€€€€€€€ОО ∆
(rot_equiv_pool2d/max_pooling2d/MaxPool_2MaxPool$rot_equiv_pool2d/GatherV2_2:output:0*/
_output_shapes
:€€€€€€€€€GG *
ksize
*
paddingVALID*
strides
Z
rot_equiv_pool2d/Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 Ry
rot_equiv_pool2d/Shape_3Shape!rot_equiv_conv2d/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	y
&rot_equiv_pool2d/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€{
(rot_equiv_pool2d/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€r
(rot_equiv_pool2d/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:∞
 rot_equiv_pool2d/strided_slice_3StridedSlice!rot_equiv_pool2d/Shape_3:output:0/rot_equiv_pool2d/strided_slice_3/stack:output:01rot_equiv_pool2d/strided_slice_3/stack_1:output:01rot_equiv_pool2d/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskZ
rot_equiv_pool2d/sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 RМ
rot_equiv_pool2d/sub_3Sub)rot_equiv_pool2d/strided_slice_3:output:0!rot_equiv_pool2d/sub_3/y:output:0*
T0	*
_output_shapes
: У
(rot_equiv_pool2d/clip_by_value_3/MinimumMinimum!rot_equiv_pool2d/Const_3:output:0rot_equiv_pool2d/sub_3:z:0*
T0	*
_output_shapes
: d
"rot_equiv_pool2d/clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R І
 rot_equiv_pool2d/clip_by_value_3Maximum,rot_equiv_pool2d/clip_by_value_3/Minimum:z:0+rot_equiv_pool2d/clip_by_value_3/y:output:0*
T0	*
_output_shapes
: k
 rot_equiv_pool2d/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ъ
rot_equiv_pool2d/GatherV2_3GatherV2!rot_equiv_conv2d/BiasAdd:output:0$rot_equiv_pool2d/clip_by_value_3:z:0)rot_equiv_pool2d/GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*1
_output_shapes
:€€€€€€€€€ОО ∆
(rot_equiv_pool2d/max_pooling2d/MaxPool_3MaxPool$rot_equiv_pool2d/GatherV2_3:output:0*/
_output_shapes
:€€€€€€€€€GG *
ksize
*
paddingVALID*
strides
ƒ
rot_equiv_pool2d/stackPack/rot_equiv_pool2d/max_pooling2d/MaxPool:output:01rot_equiv_pool2d/max_pooling2d/MaxPool_1:output:01rot_equiv_pool2d/max_pooling2d/MaxPool_2:output:01rot_equiv_pool2d/max_pooling2d/MaxPool_3:output:0*
N*
T0*3
_output_shapes!
:€€€€€€€€€GG *
axisю€€€€€€€€Z
rot_equiv_conv2d_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R w
rot_equiv_conv2d_1/ShapeShaperot_equiv_pool2d/stack:output:0*
T0*
_output_shapes
:*
out_type0	y
&rot_equiv_conv2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€{
(rot_equiv_conv2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€r
(rot_equiv_conv2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:∞
 rot_equiv_conv2d_1/strided_sliceStridedSlice!rot_equiv_conv2d_1/Shape:output:0/rot_equiv_conv2d_1/strided_slice/stack:output:01rot_equiv_conv2d_1/strided_slice/stack_1:output:01rot_equiv_conv2d_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskZ
rot_equiv_conv2d_1/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RМ
rot_equiv_conv2d_1/subSub)rot_equiv_conv2d_1/strided_slice:output:0!rot_equiv_conv2d_1/sub/y:output:0*
T0	*
_output_shapes
: У
(rot_equiv_conv2d_1/clip_by_value/MinimumMinimum!rot_equiv_conv2d_1/Const:output:0rot_equiv_conv2d_1/sub:z:0*
T0	*
_output_shapes
: d
"rot_equiv_conv2d_1/clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R І
 rot_equiv_conv2d_1/clip_by_valueMaximum,rot_equiv_conv2d_1/clip_by_value/Minimum:z:0+rot_equiv_conv2d_1/clip_by_value/y:output:0*
T0	*
_output_shapes
: k
 rot_equiv_conv2d_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ц
rot_equiv_conv2d_1/GatherV2GatherV2rot_equiv_pool2d/stack:output:0$rot_equiv_conv2d_1/clip_by_value:z:0)rot_equiv_conv2d_1/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€GG ђ
-rot_equiv_conv2d_1/convolution/ReadVariableOpReadVariableOp6rot_equiv_conv2d_1_convolution_readvariableop_resource*&
_output_shapes
:  *
dtype0и
rot_equiv_conv2d_1/convolutionConv2D$rot_equiv_conv2d_1/GatherV2:output:05rot_equiv_conv2d_1/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€EE *
paddingVALID*
strides
\
rot_equiv_conv2d_1/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 Ry
rot_equiv_conv2d_1/Shape_1Shaperot_equiv_pool2d/stack:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_conv2d_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€}
*rot_equiv_conv2d_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€t
*rot_equiv_conv2d_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ї
"rot_equiv_conv2d_1/strided_slice_1StridedSlice#rot_equiv_conv2d_1/Shape_1:output:01rot_equiv_conv2d_1/strided_slice_1/stack:output:03rot_equiv_conv2d_1/strided_slice_1/stack_1:output:03rot_equiv_conv2d_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_conv2d_1/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 RТ
rot_equiv_conv2d_1/sub_1Sub+rot_equiv_conv2d_1/strided_slice_1:output:0#rot_equiv_conv2d_1/sub_1/y:output:0*
T0	*
_output_shapes
: Щ
*rot_equiv_conv2d_1/clip_by_value_1/MinimumMinimum#rot_equiv_conv2d_1/Const_1:output:0rot_equiv_conv2d_1/sub_1:z:0*
T0	*
_output_shapes
: f
$rot_equiv_conv2d_1/clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ≠
"rot_equiv_conv2d_1/clip_by_value_1Maximum.rot_equiv_conv2d_1/clip_by_value_1/Minimum:z:0-rot_equiv_conv2d_1/clip_by_value_1/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_conv2d_1/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ь
rot_equiv_conv2d_1/GatherV2_1GatherV2rot_equiv_pool2d/stack:output:0&rot_equiv_conv2d_1/clip_by_value_1:z:0+rot_equiv_conv2d_1/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€GG Ѓ
/rot_equiv_conv2d_1/convolution_1/ReadVariableOpReadVariableOp6rot_equiv_conv2d_1_convolution_readvariableop_resource*&
_output_shapes
:  *
dtype0о
 rot_equiv_conv2d_1/convolution_1Conv2D&rot_equiv_conv2d_1/GatherV2_1:output:07rot_equiv_conv2d_1/convolution_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€EE *
paddingVALID*
strides
\
rot_equiv_conv2d_1/Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 Ry
rot_equiv_conv2d_1/Shape_2Shaperot_equiv_pool2d/stack:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_conv2d_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€}
*rot_equiv_conv2d_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€t
*rot_equiv_conv2d_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ї
"rot_equiv_conv2d_1/strided_slice_2StridedSlice#rot_equiv_conv2d_1/Shape_2:output:01rot_equiv_conv2d_1/strided_slice_2/stack:output:03rot_equiv_conv2d_1/strided_slice_2/stack_1:output:03rot_equiv_conv2d_1/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_conv2d_1/sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 RТ
rot_equiv_conv2d_1/sub_2Sub+rot_equiv_conv2d_1/strided_slice_2:output:0#rot_equiv_conv2d_1/sub_2/y:output:0*
T0	*
_output_shapes
: Щ
*rot_equiv_conv2d_1/clip_by_value_2/MinimumMinimum#rot_equiv_conv2d_1/Const_2:output:0rot_equiv_conv2d_1/sub_2:z:0*
T0	*
_output_shapes
: f
$rot_equiv_conv2d_1/clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ≠
"rot_equiv_conv2d_1/clip_by_value_2Maximum.rot_equiv_conv2d_1/clip_by_value_2/Minimum:z:0-rot_equiv_conv2d_1/clip_by_value_2/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_conv2d_1/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ь
rot_equiv_conv2d_1/GatherV2_2GatherV2rot_equiv_pool2d/stack:output:0&rot_equiv_conv2d_1/clip_by_value_2:z:0+rot_equiv_conv2d_1/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€GG Ѓ
/rot_equiv_conv2d_1/convolution_2/ReadVariableOpReadVariableOp6rot_equiv_conv2d_1_convolution_readvariableop_resource*&
_output_shapes
:  *
dtype0о
 rot_equiv_conv2d_1/convolution_2Conv2D&rot_equiv_conv2d_1/GatherV2_2:output:07rot_equiv_conv2d_1/convolution_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€EE *
paddingVALID*
strides
\
rot_equiv_conv2d_1/Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 Ry
rot_equiv_conv2d_1/Shape_3Shaperot_equiv_pool2d/stack:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_conv2d_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€}
*rot_equiv_conv2d_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€t
*rot_equiv_conv2d_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ї
"rot_equiv_conv2d_1/strided_slice_3StridedSlice#rot_equiv_conv2d_1/Shape_3:output:01rot_equiv_conv2d_1/strided_slice_3/stack:output:03rot_equiv_conv2d_1/strided_slice_3/stack_1:output:03rot_equiv_conv2d_1/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_conv2d_1/sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 RТ
rot_equiv_conv2d_1/sub_3Sub+rot_equiv_conv2d_1/strided_slice_3:output:0#rot_equiv_conv2d_1/sub_3/y:output:0*
T0	*
_output_shapes
: Щ
*rot_equiv_conv2d_1/clip_by_value_3/MinimumMinimum#rot_equiv_conv2d_1/Const_3:output:0rot_equiv_conv2d_1/sub_3:z:0*
T0	*
_output_shapes
: f
$rot_equiv_conv2d_1/clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ≠
"rot_equiv_conv2d_1/clip_by_value_3Maximum.rot_equiv_conv2d_1/clip_by_value_3/Minimum:z:0-rot_equiv_conv2d_1/clip_by_value_3/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_conv2d_1/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ь
rot_equiv_conv2d_1/GatherV2_3GatherV2rot_equiv_pool2d/stack:output:0&rot_equiv_conv2d_1/clip_by_value_3:z:0+rot_equiv_conv2d_1/GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€GG Ѓ
/rot_equiv_conv2d_1/convolution_3/ReadVariableOpReadVariableOp6rot_equiv_conv2d_1_convolution_readvariableop_resource*&
_output_shapes
:  *
dtype0о
 rot_equiv_conv2d_1/convolution_3Conv2D&rot_equiv_conv2d_1/GatherV2_3:output:07rot_equiv_conv2d_1/convolution_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€EE *
paddingVALID*
strides
¶
rot_equiv_conv2d_1/stackPack'rot_equiv_conv2d_1/convolution:output:0)rot_equiv_conv2d_1/convolution_1:output:0)rot_equiv_conv2d_1/convolution_2:output:0)rot_equiv_conv2d_1/convolution_3:output:0*
N*
T0*3
_output_shapes!
:€€€€€€€€€EE *
axisю€€€€€€€€А
rot_equiv_conv2d_1/ReluRelu!rot_equiv_conv2d_1/stack:output:0*
T0*3
_output_shapes!
:€€€€€€€€€EE Ш
)rot_equiv_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp2rot_equiv_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0љ
rot_equiv_conv2d_1/BiasAddBiasAdd%rot_equiv_conv2d_1/Relu:activations:01rot_equiv_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€EE Z
rot_equiv_pool2d_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R {
rot_equiv_pool2d_1/ShapeShape#rot_equiv_conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	y
&rot_equiv_pool2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€{
(rot_equiv_pool2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€r
(rot_equiv_pool2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:∞
 rot_equiv_pool2d_1/strided_sliceStridedSlice!rot_equiv_pool2d_1/Shape:output:0/rot_equiv_pool2d_1/strided_slice/stack:output:01rot_equiv_pool2d_1/strided_slice/stack_1:output:01rot_equiv_pool2d_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskZ
rot_equiv_pool2d_1/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RМ
rot_equiv_pool2d_1/subSub)rot_equiv_pool2d_1/strided_slice:output:0!rot_equiv_pool2d_1/sub/y:output:0*
T0	*
_output_shapes
: У
(rot_equiv_pool2d_1/clip_by_value/MinimumMinimum!rot_equiv_pool2d_1/Const:output:0rot_equiv_pool2d_1/sub:z:0*
T0	*
_output_shapes
: d
"rot_equiv_pool2d_1/clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R І
 rot_equiv_pool2d_1/clip_by_valueMaximum,rot_equiv_pool2d_1/clip_by_value/Minimum:z:0+rot_equiv_pool2d_1/clip_by_value/y:output:0*
T0	*
_output_shapes
: k
 rot_equiv_pool2d_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ъ
rot_equiv_pool2d_1/GatherV2GatherV2#rot_equiv_conv2d_1/BiasAdd:output:0$rot_equiv_pool2d_1/clip_by_value:z:0)rot_equiv_pool2d_1/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€EE »
*rot_equiv_pool2d_1/max_pooling2d_1/MaxPoolMaxPool$rot_equiv_pool2d_1/GatherV2:output:0*/
_output_shapes
:€€€€€€€€€"" *
ksize
*
paddingVALID*
strides
\
rot_equiv_pool2d_1/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 R}
rot_equiv_pool2d_1/Shape_1Shape#rot_equiv_conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_pool2d_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€}
*rot_equiv_pool2d_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€t
*rot_equiv_pool2d_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ї
"rot_equiv_pool2d_1/strided_slice_1StridedSlice#rot_equiv_pool2d_1/Shape_1:output:01rot_equiv_pool2d_1/strided_slice_1/stack:output:03rot_equiv_pool2d_1/strided_slice_1/stack_1:output:03rot_equiv_pool2d_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_pool2d_1/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 RТ
rot_equiv_pool2d_1/sub_1Sub+rot_equiv_pool2d_1/strided_slice_1:output:0#rot_equiv_pool2d_1/sub_1/y:output:0*
T0	*
_output_shapes
: Щ
*rot_equiv_pool2d_1/clip_by_value_1/MinimumMinimum#rot_equiv_pool2d_1/Const_1:output:0rot_equiv_pool2d_1/sub_1:z:0*
T0	*
_output_shapes
: f
$rot_equiv_pool2d_1/clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ≠
"rot_equiv_pool2d_1/clip_by_value_1Maximum.rot_equiv_pool2d_1/clip_by_value_1/Minimum:z:0-rot_equiv_pool2d_1/clip_by_value_1/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_pool2d_1/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€А
rot_equiv_pool2d_1/GatherV2_1GatherV2#rot_equiv_conv2d_1/BiasAdd:output:0&rot_equiv_pool2d_1/clip_by_value_1:z:0+rot_equiv_pool2d_1/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€EE ћ
,rot_equiv_pool2d_1/max_pooling2d_1/MaxPool_1MaxPool&rot_equiv_pool2d_1/GatherV2_1:output:0*/
_output_shapes
:€€€€€€€€€"" *
ksize
*
paddingVALID*
strides
\
rot_equiv_pool2d_1/Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R}
rot_equiv_pool2d_1/Shape_2Shape#rot_equiv_conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_pool2d_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€}
*rot_equiv_pool2d_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€t
*rot_equiv_pool2d_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ї
"rot_equiv_pool2d_1/strided_slice_2StridedSlice#rot_equiv_pool2d_1/Shape_2:output:01rot_equiv_pool2d_1/strided_slice_2/stack:output:03rot_equiv_pool2d_1/strided_slice_2/stack_1:output:03rot_equiv_pool2d_1/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_pool2d_1/sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 RТ
rot_equiv_pool2d_1/sub_2Sub+rot_equiv_pool2d_1/strided_slice_2:output:0#rot_equiv_pool2d_1/sub_2/y:output:0*
T0	*
_output_shapes
: Щ
*rot_equiv_pool2d_1/clip_by_value_2/MinimumMinimum#rot_equiv_pool2d_1/Const_2:output:0rot_equiv_pool2d_1/sub_2:z:0*
T0	*
_output_shapes
: f
$rot_equiv_pool2d_1/clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ≠
"rot_equiv_pool2d_1/clip_by_value_2Maximum.rot_equiv_pool2d_1/clip_by_value_2/Minimum:z:0-rot_equiv_pool2d_1/clip_by_value_2/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_pool2d_1/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€А
rot_equiv_pool2d_1/GatherV2_2GatherV2#rot_equiv_conv2d_1/BiasAdd:output:0&rot_equiv_pool2d_1/clip_by_value_2:z:0+rot_equiv_pool2d_1/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€EE ћ
,rot_equiv_pool2d_1/max_pooling2d_1/MaxPool_2MaxPool&rot_equiv_pool2d_1/GatherV2_2:output:0*/
_output_shapes
:€€€€€€€€€"" *
ksize
*
paddingVALID*
strides
\
rot_equiv_pool2d_1/Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 R}
rot_equiv_pool2d_1/Shape_3Shape#rot_equiv_conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_pool2d_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€}
*rot_equiv_pool2d_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€t
*rot_equiv_pool2d_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ї
"rot_equiv_pool2d_1/strided_slice_3StridedSlice#rot_equiv_pool2d_1/Shape_3:output:01rot_equiv_pool2d_1/strided_slice_3/stack:output:03rot_equiv_pool2d_1/strided_slice_3/stack_1:output:03rot_equiv_pool2d_1/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_pool2d_1/sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 RТ
rot_equiv_pool2d_1/sub_3Sub+rot_equiv_pool2d_1/strided_slice_3:output:0#rot_equiv_pool2d_1/sub_3/y:output:0*
T0	*
_output_shapes
: Щ
*rot_equiv_pool2d_1/clip_by_value_3/MinimumMinimum#rot_equiv_pool2d_1/Const_3:output:0rot_equiv_pool2d_1/sub_3:z:0*
T0	*
_output_shapes
: f
$rot_equiv_pool2d_1/clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ≠
"rot_equiv_pool2d_1/clip_by_value_3Maximum.rot_equiv_pool2d_1/clip_by_value_3/Minimum:z:0-rot_equiv_pool2d_1/clip_by_value_3/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_pool2d_1/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€А
rot_equiv_pool2d_1/GatherV2_3GatherV2#rot_equiv_conv2d_1/BiasAdd:output:0&rot_equiv_pool2d_1/clip_by_value_3:z:0+rot_equiv_pool2d_1/GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€EE ћ
,rot_equiv_pool2d_1/max_pooling2d_1/MaxPool_3MaxPool&rot_equiv_pool2d_1/GatherV2_3:output:0*/
_output_shapes
:€€€€€€€€€"" *
ksize
*
paddingVALID*
strides
÷
rot_equiv_pool2d_1/stackPack3rot_equiv_pool2d_1/max_pooling2d_1/MaxPool:output:05rot_equiv_pool2d_1/max_pooling2d_1/MaxPool_1:output:05rot_equiv_pool2d_1/max_pooling2d_1/MaxPool_2:output:05rot_equiv_pool2d_1/max_pooling2d_1/MaxPool_3:output:0*
N*
T0*3
_output_shapes!
:€€€€€€€€€"" *
axisю€€€€€€€€Z
rot_equiv_conv2d_2/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R y
rot_equiv_conv2d_2/ShapeShape!rot_equiv_pool2d_1/stack:output:0*
T0*
_output_shapes
:*
out_type0	y
&rot_equiv_conv2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€{
(rot_equiv_conv2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€r
(rot_equiv_conv2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:∞
 rot_equiv_conv2d_2/strided_sliceStridedSlice!rot_equiv_conv2d_2/Shape:output:0/rot_equiv_conv2d_2/strided_slice/stack:output:01rot_equiv_conv2d_2/strided_slice/stack_1:output:01rot_equiv_conv2d_2/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskZ
rot_equiv_conv2d_2/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RМ
rot_equiv_conv2d_2/subSub)rot_equiv_conv2d_2/strided_slice:output:0!rot_equiv_conv2d_2/sub/y:output:0*
T0	*
_output_shapes
: У
(rot_equiv_conv2d_2/clip_by_value/MinimumMinimum!rot_equiv_conv2d_2/Const:output:0rot_equiv_conv2d_2/sub:z:0*
T0	*
_output_shapes
: d
"rot_equiv_conv2d_2/clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R І
 rot_equiv_conv2d_2/clip_by_valueMaximum,rot_equiv_conv2d_2/clip_by_value/Minimum:z:0+rot_equiv_conv2d_2/clip_by_value/y:output:0*
T0	*
_output_shapes
: k
 rot_equiv_conv2d_2/GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ш
rot_equiv_conv2d_2/GatherV2GatherV2!rot_equiv_pool2d_1/stack:output:0$rot_equiv_conv2d_2/clip_by_value:z:0)rot_equiv_conv2d_2/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€"" ђ
-rot_equiv_conv2d_2/convolution/ReadVariableOpReadVariableOp6rot_equiv_conv2d_2_convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0и
rot_equiv_conv2d_2/convolutionConv2D$rot_equiv_conv2d_2/GatherV2:output:05rot_equiv_conv2d_2/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  @*
paddingVALID*
strides
\
rot_equiv_conv2d_2/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 R{
rot_equiv_conv2d_2/Shape_1Shape!rot_equiv_pool2d_1/stack:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_conv2d_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€}
*rot_equiv_conv2d_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€t
*rot_equiv_conv2d_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ї
"rot_equiv_conv2d_2/strided_slice_1StridedSlice#rot_equiv_conv2d_2/Shape_1:output:01rot_equiv_conv2d_2/strided_slice_1/stack:output:03rot_equiv_conv2d_2/strided_slice_1/stack_1:output:03rot_equiv_conv2d_2/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_conv2d_2/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 RТ
rot_equiv_conv2d_2/sub_1Sub+rot_equiv_conv2d_2/strided_slice_1:output:0#rot_equiv_conv2d_2/sub_1/y:output:0*
T0	*
_output_shapes
: Щ
*rot_equiv_conv2d_2/clip_by_value_1/MinimumMinimum#rot_equiv_conv2d_2/Const_1:output:0rot_equiv_conv2d_2/sub_1:z:0*
T0	*
_output_shapes
: f
$rot_equiv_conv2d_2/clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ≠
"rot_equiv_conv2d_2/clip_by_value_1Maximum.rot_equiv_conv2d_2/clip_by_value_1/Minimum:z:0-rot_equiv_conv2d_2/clip_by_value_1/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_conv2d_2/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ю
rot_equiv_conv2d_2/GatherV2_1GatherV2!rot_equiv_pool2d_1/stack:output:0&rot_equiv_conv2d_2/clip_by_value_1:z:0+rot_equiv_conv2d_2/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€"" Ѓ
/rot_equiv_conv2d_2/convolution_1/ReadVariableOpReadVariableOp6rot_equiv_conv2d_2_convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0о
 rot_equiv_conv2d_2/convolution_1Conv2D&rot_equiv_conv2d_2/GatherV2_1:output:07rot_equiv_conv2d_2/convolution_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  @*
paddingVALID*
strides
\
rot_equiv_conv2d_2/Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R{
rot_equiv_conv2d_2/Shape_2Shape!rot_equiv_pool2d_1/stack:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_conv2d_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€}
*rot_equiv_conv2d_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€t
*rot_equiv_conv2d_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ї
"rot_equiv_conv2d_2/strided_slice_2StridedSlice#rot_equiv_conv2d_2/Shape_2:output:01rot_equiv_conv2d_2/strided_slice_2/stack:output:03rot_equiv_conv2d_2/strided_slice_2/stack_1:output:03rot_equiv_conv2d_2/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_conv2d_2/sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 RТ
rot_equiv_conv2d_2/sub_2Sub+rot_equiv_conv2d_2/strided_slice_2:output:0#rot_equiv_conv2d_2/sub_2/y:output:0*
T0	*
_output_shapes
: Щ
*rot_equiv_conv2d_2/clip_by_value_2/MinimumMinimum#rot_equiv_conv2d_2/Const_2:output:0rot_equiv_conv2d_2/sub_2:z:0*
T0	*
_output_shapes
: f
$rot_equiv_conv2d_2/clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ≠
"rot_equiv_conv2d_2/clip_by_value_2Maximum.rot_equiv_conv2d_2/clip_by_value_2/Minimum:z:0-rot_equiv_conv2d_2/clip_by_value_2/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_conv2d_2/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ю
rot_equiv_conv2d_2/GatherV2_2GatherV2!rot_equiv_pool2d_1/stack:output:0&rot_equiv_conv2d_2/clip_by_value_2:z:0+rot_equiv_conv2d_2/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€"" Ѓ
/rot_equiv_conv2d_2/convolution_2/ReadVariableOpReadVariableOp6rot_equiv_conv2d_2_convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0о
 rot_equiv_conv2d_2/convolution_2Conv2D&rot_equiv_conv2d_2/GatherV2_2:output:07rot_equiv_conv2d_2/convolution_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  @*
paddingVALID*
strides
\
rot_equiv_conv2d_2/Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 R{
rot_equiv_conv2d_2/Shape_3Shape!rot_equiv_pool2d_1/stack:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_conv2d_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€}
*rot_equiv_conv2d_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€t
*rot_equiv_conv2d_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ї
"rot_equiv_conv2d_2/strided_slice_3StridedSlice#rot_equiv_conv2d_2/Shape_3:output:01rot_equiv_conv2d_2/strided_slice_3/stack:output:03rot_equiv_conv2d_2/strided_slice_3/stack_1:output:03rot_equiv_conv2d_2/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_conv2d_2/sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 RТ
rot_equiv_conv2d_2/sub_3Sub+rot_equiv_conv2d_2/strided_slice_3:output:0#rot_equiv_conv2d_2/sub_3/y:output:0*
T0	*
_output_shapes
: Щ
*rot_equiv_conv2d_2/clip_by_value_3/MinimumMinimum#rot_equiv_conv2d_2/Const_3:output:0rot_equiv_conv2d_2/sub_3:z:0*
T0	*
_output_shapes
: f
$rot_equiv_conv2d_2/clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ≠
"rot_equiv_conv2d_2/clip_by_value_3Maximum.rot_equiv_conv2d_2/clip_by_value_3/Minimum:z:0-rot_equiv_conv2d_2/clip_by_value_3/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_conv2d_2/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ю
rot_equiv_conv2d_2/GatherV2_3GatherV2!rot_equiv_pool2d_1/stack:output:0&rot_equiv_conv2d_2/clip_by_value_3:z:0+rot_equiv_conv2d_2/GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€"" Ѓ
/rot_equiv_conv2d_2/convolution_3/ReadVariableOpReadVariableOp6rot_equiv_conv2d_2_convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0о
 rot_equiv_conv2d_2/convolution_3Conv2D&rot_equiv_conv2d_2/GatherV2_3:output:07rot_equiv_conv2d_2/convolution_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  @*
paddingVALID*
strides
¶
rot_equiv_conv2d_2/stackPack'rot_equiv_conv2d_2/convolution:output:0)rot_equiv_conv2d_2/convolution_1:output:0)rot_equiv_conv2d_2/convolution_2:output:0)rot_equiv_conv2d_2/convolution_3:output:0*
N*
T0*3
_output_shapes!
:€€€€€€€€€  @*
axisю€€€€€€€€А
rot_equiv_conv2d_2/ReluRelu!rot_equiv_conv2d_2/stack:output:0*
T0*3
_output_shapes!
:€€€€€€€€€  @Ш
)rot_equiv_conv2d_2/BiasAdd/ReadVariableOpReadVariableOp2rot_equiv_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0љ
rot_equiv_conv2d_2/BiasAddBiasAdd%rot_equiv_conv2d_2/Relu:activations:01rot_equiv_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€  @Z
rot_equiv_pool2d_2/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R {
rot_equiv_pool2d_2/ShapeShape#rot_equiv_conv2d_2/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	y
&rot_equiv_pool2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€{
(rot_equiv_pool2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€r
(rot_equiv_pool2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:∞
 rot_equiv_pool2d_2/strided_sliceStridedSlice!rot_equiv_pool2d_2/Shape:output:0/rot_equiv_pool2d_2/strided_slice/stack:output:01rot_equiv_pool2d_2/strided_slice/stack_1:output:01rot_equiv_pool2d_2/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskZ
rot_equiv_pool2d_2/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RМ
rot_equiv_pool2d_2/subSub)rot_equiv_pool2d_2/strided_slice:output:0!rot_equiv_pool2d_2/sub/y:output:0*
T0	*
_output_shapes
: У
(rot_equiv_pool2d_2/clip_by_value/MinimumMinimum!rot_equiv_pool2d_2/Const:output:0rot_equiv_pool2d_2/sub:z:0*
T0	*
_output_shapes
: d
"rot_equiv_pool2d_2/clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R І
 rot_equiv_pool2d_2/clip_by_valueMaximum,rot_equiv_pool2d_2/clip_by_value/Minimum:z:0+rot_equiv_pool2d_2/clip_by_value/y:output:0*
T0	*
_output_shapes
: k
 rot_equiv_pool2d_2/GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ъ
rot_equiv_pool2d_2/GatherV2GatherV2#rot_equiv_conv2d_2/BiasAdd:output:0$rot_equiv_pool2d_2/clip_by_value:z:0)rot_equiv_pool2d_2/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€  @»
*rot_equiv_pool2d_2/max_pooling2d_2/MaxPoolMaxPool$rot_equiv_pool2d_2/GatherV2:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
\
rot_equiv_pool2d_2/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 R}
rot_equiv_pool2d_2/Shape_1Shape#rot_equiv_conv2d_2/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_pool2d_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€}
*rot_equiv_pool2d_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€t
*rot_equiv_pool2d_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ї
"rot_equiv_pool2d_2/strided_slice_1StridedSlice#rot_equiv_pool2d_2/Shape_1:output:01rot_equiv_pool2d_2/strided_slice_1/stack:output:03rot_equiv_pool2d_2/strided_slice_1/stack_1:output:03rot_equiv_pool2d_2/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_pool2d_2/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 RТ
rot_equiv_pool2d_2/sub_1Sub+rot_equiv_pool2d_2/strided_slice_1:output:0#rot_equiv_pool2d_2/sub_1/y:output:0*
T0	*
_output_shapes
: Щ
*rot_equiv_pool2d_2/clip_by_value_1/MinimumMinimum#rot_equiv_pool2d_2/Const_1:output:0rot_equiv_pool2d_2/sub_1:z:0*
T0	*
_output_shapes
: f
$rot_equiv_pool2d_2/clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ≠
"rot_equiv_pool2d_2/clip_by_value_1Maximum.rot_equiv_pool2d_2/clip_by_value_1/Minimum:z:0-rot_equiv_pool2d_2/clip_by_value_1/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_pool2d_2/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€А
rot_equiv_pool2d_2/GatherV2_1GatherV2#rot_equiv_conv2d_2/BiasAdd:output:0&rot_equiv_pool2d_2/clip_by_value_1:z:0+rot_equiv_pool2d_2/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€  @ћ
,rot_equiv_pool2d_2/max_pooling2d_2/MaxPool_1MaxPool&rot_equiv_pool2d_2/GatherV2_1:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
\
rot_equiv_pool2d_2/Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R}
rot_equiv_pool2d_2/Shape_2Shape#rot_equiv_conv2d_2/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_pool2d_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€}
*rot_equiv_pool2d_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€t
*rot_equiv_pool2d_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ї
"rot_equiv_pool2d_2/strided_slice_2StridedSlice#rot_equiv_pool2d_2/Shape_2:output:01rot_equiv_pool2d_2/strided_slice_2/stack:output:03rot_equiv_pool2d_2/strided_slice_2/stack_1:output:03rot_equiv_pool2d_2/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_pool2d_2/sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 RТ
rot_equiv_pool2d_2/sub_2Sub+rot_equiv_pool2d_2/strided_slice_2:output:0#rot_equiv_pool2d_2/sub_2/y:output:0*
T0	*
_output_shapes
: Щ
*rot_equiv_pool2d_2/clip_by_value_2/MinimumMinimum#rot_equiv_pool2d_2/Const_2:output:0rot_equiv_pool2d_2/sub_2:z:0*
T0	*
_output_shapes
: f
$rot_equiv_pool2d_2/clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ≠
"rot_equiv_pool2d_2/clip_by_value_2Maximum.rot_equiv_pool2d_2/clip_by_value_2/Minimum:z:0-rot_equiv_pool2d_2/clip_by_value_2/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_pool2d_2/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€А
rot_equiv_pool2d_2/GatherV2_2GatherV2#rot_equiv_conv2d_2/BiasAdd:output:0&rot_equiv_pool2d_2/clip_by_value_2:z:0+rot_equiv_pool2d_2/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€  @ћ
,rot_equiv_pool2d_2/max_pooling2d_2/MaxPool_2MaxPool&rot_equiv_pool2d_2/GatherV2_2:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
\
rot_equiv_pool2d_2/Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 R}
rot_equiv_pool2d_2/Shape_3Shape#rot_equiv_conv2d_2/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_pool2d_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€}
*rot_equiv_pool2d_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€t
*rot_equiv_pool2d_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ї
"rot_equiv_pool2d_2/strided_slice_3StridedSlice#rot_equiv_pool2d_2/Shape_3:output:01rot_equiv_pool2d_2/strided_slice_3/stack:output:03rot_equiv_pool2d_2/strided_slice_3/stack_1:output:03rot_equiv_pool2d_2/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_pool2d_2/sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 RТ
rot_equiv_pool2d_2/sub_3Sub+rot_equiv_pool2d_2/strided_slice_3:output:0#rot_equiv_pool2d_2/sub_3/y:output:0*
T0	*
_output_shapes
: Щ
*rot_equiv_pool2d_2/clip_by_value_3/MinimumMinimum#rot_equiv_pool2d_2/Const_3:output:0rot_equiv_pool2d_2/sub_3:z:0*
T0	*
_output_shapes
: f
$rot_equiv_pool2d_2/clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ≠
"rot_equiv_pool2d_2/clip_by_value_3Maximum.rot_equiv_pool2d_2/clip_by_value_3/Minimum:z:0-rot_equiv_pool2d_2/clip_by_value_3/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_pool2d_2/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€А
rot_equiv_pool2d_2/GatherV2_3GatherV2#rot_equiv_conv2d_2/BiasAdd:output:0&rot_equiv_pool2d_2/clip_by_value_3:z:0+rot_equiv_pool2d_2/GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€  @ћ
,rot_equiv_pool2d_2/max_pooling2d_2/MaxPool_3MaxPool&rot_equiv_pool2d_2/GatherV2_3:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
÷
rot_equiv_pool2d_2/stackPack3rot_equiv_pool2d_2/max_pooling2d_2/MaxPool:output:05rot_equiv_pool2d_2/max_pooling2d_2/MaxPool_1:output:05rot_equiv_pool2d_2/max_pooling2d_2/MaxPool_2:output:05rot_equiv_pool2d_2/max_pooling2d_2/MaxPool_3:output:0*
N*
T0*3
_output_shapes!
:€€€€€€€€€@*
axisю€€€€€€€€Z
rot_equiv_conv2d_3/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R y
rot_equiv_conv2d_3/ShapeShape!rot_equiv_pool2d_2/stack:output:0*
T0*
_output_shapes
:*
out_type0	y
&rot_equiv_conv2d_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€{
(rot_equiv_conv2d_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€r
(rot_equiv_conv2d_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:∞
 rot_equiv_conv2d_3/strided_sliceStridedSlice!rot_equiv_conv2d_3/Shape:output:0/rot_equiv_conv2d_3/strided_slice/stack:output:01rot_equiv_conv2d_3/strided_slice/stack_1:output:01rot_equiv_conv2d_3/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskZ
rot_equiv_conv2d_3/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RМ
rot_equiv_conv2d_3/subSub)rot_equiv_conv2d_3/strided_slice:output:0!rot_equiv_conv2d_3/sub/y:output:0*
T0	*
_output_shapes
: У
(rot_equiv_conv2d_3/clip_by_value/MinimumMinimum!rot_equiv_conv2d_3/Const:output:0rot_equiv_conv2d_3/sub:z:0*
T0	*
_output_shapes
: d
"rot_equiv_conv2d_3/clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R І
 rot_equiv_conv2d_3/clip_by_valueMaximum,rot_equiv_conv2d_3/clip_by_value/Minimum:z:0+rot_equiv_conv2d_3/clip_by_value/y:output:0*
T0	*
_output_shapes
: k
 rot_equiv_conv2d_3/GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ш
rot_equiv_conv2d_3/GatherV2GatherV2!rot_equiv_pool2d_2/stack:output:0$rot_equiv_conv2d_3/clip_by_value:z:0)rot_equiv_conv2d_3/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@ђ
-rot_equiv_conv2d_3/convolution/ReadVariableOpReadVariableOp6rot_equiv_conv2d_3_convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype0и
rot_equiv_conv2d_3/convolutionConv2D$rot_equiv_conv2d_3/GatherV2:output:05rot_equiv_conv2d_3/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingVALID*
strides
\
rot_equiv_conv2d_3/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 R{
rot_equiv_conv2d_3/Shape_1Shape!rot_equiv_pool2d_2/stack:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_conv2d_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€}
*rot_equiv_conv2d_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€t
*rot_equiv_conv2d_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ї
"rot_equiv_conv2d_3/strided_slice_1StridedSlice#rot_equiv_conv2d_3/Shape_1:output:01rot_equiv_conv2d_3/strided_slice_1/stack:output:03rot_equiv_conv2d_3/strided_slice_1/stack_1:output:03rot_equiv_conv2d_3/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_conv2d_3/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 RТ
rot_equiv_conv2d_3/sub_1Sub+rot_equiv_conv2d_3/strided_slice_1:output:0#rot_equiv_conv2d_3/sub_1/y:output:0*
T0	*
_output_shapes
: Щ
*rot_equiv_conv2d_3/clip_by_value_1/MinimumMinimum#rot_equiv_conv2d_3/Const_1:output:0rot_equiv_conv2d_3/sub_1:z:0*
T0	*
_output_shapes
: f
$rot_equiv_conv2d_3/clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ≠
"rot_equiv_conv2d_3/clip_by_value_1Maximum.rot_equiv_conv2d_3/clip_by_value_1/Minimum:z:0-rot_equiv_conv2d_3/clip_by_value_1/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_conv2d_3/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ю
rot_equiv_conv2d_3/GatherV2_1GatherV2!rot_equiv_pool2d_2/stack:output:0&rot_equiv_conv2d_3/clip_by_value_1:z:0+rot_equiv_conv2d_3/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@Ѓ
/rot_equiv_conv2d_3/convolution_1/ReadVariableOpReadVariableOp6rot_equiv_conv2d_3_convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype0о
 rot_equiv_conv2d_3/convolution_1Conv2D&rot_equiv_conv2d_3/GatherV2_1:output:07rot_equiv_conv2d_3/convolution_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingVALID*
strides
\
rot_equiv_conv2d_3/Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R{
rot_equiv_conv2d_3/Shape_2Shape!rot_equiv_pool2d_2/stack:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_conv2d_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€}
*rot_equiv_conv2d_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€t
*rot_equiv_conv2d_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ї
"rot_equiv_conv2d_3/strided_slice_2StridedSlice#rot_equiv_conv2d_3/Shape_2:output:01rot_equiv_conv2d_3/strided_slice_2/stack:output:03rot_equiv_conv2d_3/strided_slice_2/stack_1:output:03rot_equiv_conv2d_3/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_conv2d_3/sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 RТ
rot_equiv_conv2d_3/sub_2Sub+rot_equiv_conv2d_3/strided_slice_2:output:0#rot_equiv_conv2d_3/sub_2/y:output:0*
T0	*
_output_shapes
: Щ
*rot_equiv_conv2d_3/clip_by_value_2/MinimumMinimum#rot_equiv_conv2d_3/Const_2:output:0rot_equiv_conv2d_3/sub_2:z:0*
T0	*
_output_shapes
: f
$rot_equiv_conv2d_3/clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ≠
"rot_equiv_conv2d_3/clip_by_value_2Maximum.rot_equiv_conv2d_3/clip_by_value_2/Minimum:z:0-rot_equiv_conv2d_3/clip_by_value_2/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_conv2d_3/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ю
rot_equiv_conv2d_3/GatherV2_2GatherV2!rot_equiv_pool2d_2/stack:output:0&rot_equiv_conv2d_3/clip_by_value_2:z:0+rot_equiv_conv2d_3/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@Ѓ
/rot_equiv_conv2d_3/convolution_2/ReadVariableOpReadVariableOp6rot_equiv_conv2d_3_convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype0о
 rot_equiv_conv2d_3/convolution_2Conv2D&rot_equiv_conv2d_3/GatherV2_2:output:07rot_equiv_conv2d_3/convolution_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingVALID*
strides
\
rot_equiv_conv2d_3/Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 R{
rot_equiv_conv2d_3/Shape_3Shape!rot_equiv_pool2d_2/stack:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_conv2d_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€}
*rot_equiv_conv2d_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€t
*rot_equiv_conv2d_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ї
"rot_equiv_conv2d_3/strided_slice_3StridedSlice#rot_equiv_conv2d_3/Shape_3:output:01rot_equiv_conv2d_3/strided_slice_3/stack:output:03rot_equiv_conv2d_3/strided_slice_3/stack_1:output:03rot_equiv_conv2d_3/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_conv2d_3/sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 RТ
rot_equiv_conv2d_3/sub_3Sub+rot_equiv_conv2d_3/strided_slice_3:output:0#rot_equiv_conv2d_3/sub_3/y:output:0*
T0	*
_output_shapes
: Щ
*rot_equiv_conv2d_3/clip_by_value_3/MinimumMinimum#rot_equiv_conv2d_3/Const_3:output:0rot_equiv_conv2d_3/sub_3:z:0*
T0	*
_output_shapes
: f
$rot_equiv_conv2d_3/clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ≠
"rot_equiv_conv2d_3/clip_by_value_3Maximum.rot_equiv_conv2d_3/clip_by_value_3/Minimum:z:0-rot_equiv_conv2d_3/clip_by_value_3/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_conv2d_3/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ю
rot_equiv_conv2d_3/GatherV2_3GatherV2!rot_equiv_pool2d_2/stack:output:0&rot_equiv_conv2d_3/clip_by_value_3:z:0+rot_equiv_conv2d_3/GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@Ѓ
/rot_equiv_conv2d_3/convolution_3/ReadVariableOpReadVariableOp6rot_equiv_conv2d_3_convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype0о
 rot_equiv_conv2d_3/convolution_3Conv2D&rot_equiv_conv2d_3/GatherV2_3:output:07rot_equiv_conv2d_3/convolution_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingVALID*
strides
¶
rot_equiv_conv2d_3/stackPack'rot_equiv_conv2d_3/convolution:output:0)rot_equiv_conv2d_3/convolution_1:output:0)rot_equiv_conv2d_3/convolution_2:output:0)rot_equiv_conv2d_3/convolution_3:output:0*
N*
T0*3
_output_shapes!
:€€€€€€€€€@*
axisю€€€€€€€€А
rot_equiv_conv2d_3/ReluRelu!rot_equiv_conv2d_3/stack:output:0*
T0*3
_output_shapes!
:€€€€€€€€€@Ш
)rot_equiv_conv2d_3/BiasAdd/ReadVariableOpReadVariableOp2rot_equiv_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0љ
rot_equiv_conv2d_3/BiasAddBiasAdd%rot_equiv_conv2d_3/Relu:activations:01rot_equiv_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€@Z
rot_equiv_pool2d_3/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R {
rot_equiv_pool2d_3/ShapeShape#rot_equiv_conv2d_3/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	y
&rot_equiv_pool2d_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€{
(rot_equiv_pool2d_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€r
(rot_equiv_pool2d_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:∞
 rot_equiv_pool2d_3/strided_sliceStridedSlice!rot_equiv_pool2d_3/Shape:output:0/rot_equiv_pool2d_3/strided_slice/stack:output:01rot_equiv_pool2d_3/strided_slice/stack_1:output:01rot_equiv_pool2d_3/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskZ
rot_equiv_pool2d_3/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RМ
rot_equiv_pool2d_3/subSub)rot_equiv_pool2d_3/strided_slice:output:0!rot_equiv_pool2d_3/sub/y:output:0*
T0	*
_output_shapes
: У
(rot_equiv_pool2d_3/clip_by_value/MinimumMinimum!rot_equiv_pool2d_3/Const:output:0rot_equiv_pool2d_3/sub:z:0*
T0	*
_output_shapes
: d
"rot_equiv_pool2d_3/clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R І
 rot_equiv_pool2d_3/clip_by_valueMaximum,rot_equiv_pool2d_3/clip_by_value/Minimum:z:0+rot_equiv_pool2d_3/clip_by_value/y:output:0*
T0	*
_output_shapes
: k
 rot_equiv_pool2d_3/GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ъ
rot_equiv_pool2d_3/GatherV2GatherV2#rot_equiv_conv2d_3/BiasAdd:output:0$rot_equiv_pool2d_3/clip_by_value:z:0)rot_equiv_pool2d_3/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@»
*rot_equiv_pool2d_3/max_pooling2d_3/MaxPoolMaxPool$rot_equiv_pool2d_3/GatherV2:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
\
rot_equiv_pool2d_3/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 R}
rot_equiv_pool2d_3/Shape_1Shape#rot_equiv_conv2d_3/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_pool2d_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€}
*rot_equiv_pool2d_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€t
*rot_equiv_pool2d_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ї
"rot_equiv_pool2d_3/strided_slice_1StridedSlice#rot_equiv_pool2d_3/Shape_1:output:01rot_equiv_pool2d_3/strided_slice_1/stack:output:03rot_equiv_pool2d_3/strided_slice_1/stack_1:output:03rot_equiv_pool2d_3/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_pool2d_3/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 RТ
rot_equiv_pool2d_3/sub_1Sub+rot_equiv_pool2d_3/strided_slice_1:output:0#rot_equiv_pool2d_3/sub_1/y:output:0*
T0	*
_output_shapes
: Щ
*rot_equiv_pool2d_3/clip_by_value_1/MinimumMinimum#rot_equiv_pool2d_3/Const_1:output:0rot_equiv_pool2d_3/sub_1:z:0*
T0	*
_output_shapes
: f
$rot_equiv_pool2d_3/clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ≠
"rot_equiv_pool2d_3/clip_by_value_1Maximum.rot_equiv_pool2d_3/clip_by_value_1/Minimum:z:0-rot_equiv_pool2d_3/clip_by_value_1/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_pool2d_3/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€А
rot_equiv_pool2d_3/GatherV2_1GatherV2#rot_equiv_conv2d_3/BiasAdd:output:0&rot_equiv_pool2d_3/clip_by_value_1:z:0+rot_equiv_pool2d_3/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@ћ
,rot_equiv_pool2d_3/max_pooling2d_3/MaxPool_1MaxPool&rot_equiv_pool2d_3/GatherV2_1:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
\
rot_equiv_pool2d_3/Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R}
rot_equiv_pool2d_3/Shape_2Shape#rot_equiv_conv2d_3/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_pool2d_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€}
*rot_equiv_pool2d_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€t
*rot_equiv_pool2d_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ї
"rot_equiv_pool2d_3/strided_slice_2StridedSlice#rot_equiv_pool2d_3/Shape_2:output:01rot_equiv_pool2d_3/strided_slice_2/stack:output:03rot_equiv_pool2d_3/strided_slice_2/stack_1:output:03rot_equiv_pool2d_3/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_pool2d_3/sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 RТ
rot_equiv_pool2d_3/sub_2Sub+rot_equiv_pool2d_3/strided_slice_2:output:0#rot_equiv_pool2d_3/sub_2/y:output:0*
T0	*
_output_shapes
: Щ
*rot_equiv_pool2d_3/clip_by_value_2/MinimumMinimum#rot_equiv_pool2d_3/Const_2:output:0rot_equiv_pool2d_3/sub_2:z:0*
T0	*
_output_shapes
: f
$rot_equiv_pool2d_3/clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ≠
"rot_equiv_pool2d_3/clip_by_value_2Maximum.rot_equiv_pool2d_3/clip_by_value_2/Minimum:z:0-rot_equiv_pool2d_3/clip_by_value_2/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_pool2d_3/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€А
rot_equiv_pool2d_3/GatherV2_2GatherV2#rot_equiv_conv2d_3/BiasAdd:output:0&rot_equiv_pool2d_3/clip_by_value_2:z:0+rot_equiv_pool2d_3/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@ћ
,rot_equiv_pool2d_3/max_pooling2d_3/MaxPool_2MaxPool&rot_equiv_pool2d_3/GatherV2_2:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
\
rot_equiv_pool2d_3/Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 R}
rot_equiv_pool2d_3/Shape_3Shape#rot_equiv_conv2d_3/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_pool2d_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€}
*rot_equiv_pool2d_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€t
*rot_equiv_pool2d_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ї
"rot_equiv_pool2d_3/strided_slice_3StridedSlice#rot_equiv_pool2d_3/Shape_3:output:01rot_equiv_pool2d_3/strided_slice_3/stack:output:03rot_equiv_pool2d_3/strided_slice_3/stack_1:output:03rot_equiv_pool2d_3/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_pool2d_3/sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 RТ
rot_equiv_pool2d_3/sub_3Sub+rot_equiv_pool2d_3/strided_slice_3:output:0#rot_equiv_pool2d_3/sub_3/y:output:0*
T0	*
_output_shapes
: Щ
*rot_equiv_pool2d_3/clip_by_value_3/MinimumMinimum#rot_equiv_pool2d_3/Const_3:output:0rot_equiv_pool2d_3/sub_3:z:0*
T0	*
_output_shapes
: f
$rot_equiv_pool2d_3/clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ≠
"rot_equiv_pool2d_3/clip_by_value_3Maximum.rot_equiv_pool2d_3/clip_by_value_3/Minimum:z:0-rot_equiv_pool2d_3/clip_by_value_3/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_pool2d_3/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€А
rot_equiv_pool2d_3/GatherV2_3GatherV2#rot_equiv_conv2d_3/BiasAdd:output:0&rot_equiv_pool2d_3/clip_by_value_3:z:0+rot_equiv_pool2d_3/GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@ћ
,rot_equiv_pool2d_3/max_pooling2d_3/MaxPool_3MaxPool&rot_equiv_pool2d_3/GatherV2_3:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
÷
rot_equiv_pool2d_3/stackPack3rot_equiv_pool2d_3/max_pooling2d_3/MaxPool:output:05rot_equiv_pool2d_3/max_pooling2d_3/MaxPool_1:output:05rot_equiv_pool2d_3/max_pooling2d_3/MaxPool_2:output:05rot_equiv_pool2d_3/max_pooling2d_3/MaxPool_3:output:0*
N*
T0*3
_output_shapes!
:€€€€€€€€€@*
axisю€€€€€€€€Z
rot_equiv_conv2d_4/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R y
rot_equiv_conv2d_4/ShapeShape!rot_equiv_pool2d_3/stack:output:0*
T0*
_output_shapes
:*
out_type0	y
&rot_equiv_conv2d_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€{
(rot_equiv_conv2d_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€r
(rot_equiv_conv2d_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:∞
 rot_equiv_conv2d_4/strided_sliceStridedSlice!rot_equiv_conv2d_4/Shape:output:0/rot_equiv_conv2d_4/strided_slice/stack:output:01rot_equiv_conv2d_4/strided_slice/stack_1:output:01rot_equiv_conv2d_4/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskZ
rot_equiv_conv2d_4/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RМ
rot_equiv_conv2d_4/subSub)rot_equiv_conv2d_4/strided_slice:output:0!rot_equiv_conv2d_4/sub/y:output:0*
T0	*
_output_shapes
: У
(rot_equiv_conv2d_4/clip_by_value/MinimumMinimum!rot_equiv_conv2d_4/Const:output:0rot_equiv_conv2d_4/sub:z:0*
T0	*
_output_shapes
: d
"rot_equiv_conv2d_4/clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R І
 rot_equiv_conv2d_4/clip_by_valueMaximum,rot_equiv_conv2d_4/clip_by_value/Minimum:z:0+rot_equiv_conv2d_4/clip_by_value/y:output:0*
T0	*
_output_shapes
: k
 rot_equiv_conv2d_4/GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ш
rot_equiv_conv2d_4/GatherV2GatherV2!rot_equiv_pool2d_3/stack:output:0$rot_equiv_conv2d_4/clip_by_value:z:0)rot_equiv_conv2d_4/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@≠
-rot_equiv_conv2d_4/convolution/ReadVariableOpReadVariableOp6rot_equiv_conv2d_4_convolution_readvariableop_resource*'
_output_shapes
:@А*
dtype0й
rot_equiv_conv2d_4/convolutionConv2D$rot_equiv_conv2d_4/GatherV2:output:05rot_equiv_conv2d_4/convolution/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingVALID*
strides
\
rot_equiv_conv2d_4/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 R{
rot_equiv_conv2d_4/Shape_1Shape!rot_equiv_pool2d_3/stack:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_conv2d_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€}
*rot_equiv_conv2d_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€t
*rot_equiv_conv2d_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ї
"rot_equiv_conv2d_4/strided_slice_1StridedSlice#rot_equiv_conv2d_4/Shape_1:output:01rot_equiv_conv2d_4/strided_slice_1/stack:output:03rot_equiv_conv2d_4/strided_slice_1/stack_1:output:03rot_equiv_conv2d_4/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_conv2d_4/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 RТ
rot_equiv_conv2d_4/sub_1Sub+rot_equiv_conv2d_4/strided_slice_1:output:0#rot_equiv_conv2d_4/sub_1/y:output:0*
T0	*
_output_shapes
: Щ
*rot_equiv_conv2d_4/clip_by_value_1/MinimumMinimum#rot_equiv_conv2d_4/Const_1:output:0rot_equiv_conv2d_4/sub_1:z:0*
T0	*
_output_shapes
: f
$rot_equiv_conv2d_4/clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ≠
"rot_equiv_conv2d_4/clip_by_value_1Maximum.rot_equiv_conv2d_4/clip_by_value_1/Minimum:z:0-rot_equiv_conv2d_4/clip_by_value_1/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_conv2d_4/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ю
rot_equiv_conv2d_4/GatherV2_1GatherV2!rot_equiv_pool2d_3/stack:output:0&rot_equiv_conv2d_4/clip_by_value_1:z:0+rot_equiv_conv2d_4/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@ѓ
/rot_equiv_conv2d_4/convolution_1/ReadVariableOpReadVariableOp6rot_equiv_conv2d_4_convolution_readvariableop_resource*'
_output_shapes
:@А*
dtype0п
 rot_equiv_conv2d_4/convolution_1Conv2D&rot_equiv_conv2d_4/GatherV2_1:output:07rot_equiv_conv2d_4/convolution_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingVALID*
strides
\
rot_equiv_conv2d_4/Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R{
rot_equiv_conv2d_4/Shape_2Shape!rot_equiv_pool2d_3/stack:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_conv2d_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€}
*rot_equiv_conv2d_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€t
*rot_equiv_conv2d_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ї
"rot_equiv_conv2d_4/strided_slice_2StridedSlice#rot_equiv_conv2d_4/Shape_2:output:01rot_equiv_conv2d_4/strided_slice_2/stack:output:03rot_equiv_conv2d_4/strided_slice_2/stack_1:output:03rot_equiv_conv2d_4/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_conv2d_4/sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 RТ
rot_equiv_conv2d_4/sub_2Sub+rot_equiv_conv2d_4/strided_slice_2:output:0#rot_equiv_conv2d_4/sub_2/y:output:0*
T0	*
_output_shapes
: Щ
*rot_equiv_conv2d_4/clip_by_value_2/MinimumMinimum#rot_equiv_conv2d_4/Const_2:output:0rot_equiv_conv2d_4/sub_2:z:0*
T0	*
_output_shapes
: f
$rot_equiv_conv2d_4/clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ≠
"rot_equiv_conv2d_4/clip_by_value_2Maximum.rot_equiv_conv2d_4/clip_by_value_2/Minimum:z:0-rot_equiv_conv2d_4/clip_by_value_2/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_conv2d_4/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ю
rot_equiv_conv2d_4/GatherV2_2GatherV2!rot_equiv_pool2d_3/stack:output:0&rot_equiv_conv2d_4/clip_by_value_2:z:0+rot_equiv_conv2d_4/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@ѓ
/rot_equiv_conv2d_4/convolution_2/ReadVariableOpReadVariableOp6rot_equiv_conv2d_4_convolution_readvariableop_resource*'
_output_shapes
:@А*
dtype0п
 rot_equiv_conv2d_4/convolution_2Conv2D&rot_equiv_conv2d_4/GatherV2_2:output:07rot_equiv_conv2d_4/convolution_2/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingVALID*
strides
\
rot_equiv_conv2d_4/Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 R{
rot_equiv_conv2d_4/Shape_3Shape!rot_equiv_pool2d_3/stack:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_conv2d_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€}
*rot_equiv_conv2d_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€t
*rot_equiv_conv2d_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ї
"rot_equiv_conv2d_4/strided_slice_3StridedSlice#rot_equiv_conv2d_4/Shape_3:output:01rot_equiv_conv2d_4/strided_slice_3/stack:output:03rot_equiv_conv2d_4/strided_slice_3/stack_1:output:03rot_equiv_conv2d_4/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_conv2d_4/sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 RТ
rot_equiv_conv2d_4/sub_3Sub+rot_equiv_conv2d_4/strided_slice_3:output:0#rot_equiv_conv2d_4/sub_3/y:output:0*
T0	*
_output_shapes
: Щ
*rot_equiv_conv2d_4/clip_by_value_3/MinimumMinimum#rot_equiv_conv2d_4/Const_3:output:0rot_equiv_conv2d_4/sub_3:z:0*
T0	*
_output_shapes
: f
$rot_equiv_conv2d_4/clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ≠
"rot_equiv_conv2d_4/clip_by_value_3Maximum.rot_equiv_conv2d_4/clip_by_value_3/Minimum:z:0-rot_equiv_conv2d_4/clip_by_value_3/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_conv2d_4/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ю
rot_equiv_conv2d_4/GatherV2_3GatherV2!rot_equiv_pool2d_3/stack:output:0&rot_equiv_conv2d_4/clip_by_value_3:z:0+rot_equiv_conv2d_4/GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@ѓ
/rot_equiv_conv2d_4/convolution_3/ReadVariableOpReadVariableOp6rot_equiv_conv2d_4_convolution_readvariableop_resource*'
_output_shapes
:@А*
dtype0п
 rot_equiv_conv2d_4/convolution_3Conv2D&rot_equiv_conv2d_4/GatherV2_3:output:07rot_equiv_conv2d_4/convolution_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingVALID*
strides
І
rot_equiv_conv2d_4/stackPack'rot_equiv_conv2d_4/convolution:output:0)rot_equiv_conv2d_4/convolution_1:output:0)rot_equiv_conv2d_4/convolution_2:output:0)rot_equiv_conv2d_4/convolution_3:output:0*
N*
T0*4
_output_shapes"
 :€€€€€€€€€А*
axisю€€€€€€€€Б
rot_equiv_conv2d_4/ReluRelu!rot_equiv_conv2d_4/stack:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€АЩ
)rot_equiv_conv2d_4/BiasAdd/ReadVariableOpReadVariableOp2rot_equiv_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Њ
rot_equiv_conv2d_4/BiasAddBiasAdd%rot_equiv_conv2d_4/Relu:activations:01rot_equiv_conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аm
"rot_inv_pool/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€§
rot_inv_pool/MaxMax#rot_equiv_conv2d_4/BiasAdd:output:0+rot_inv_pool/Max/reduction_indices:output:0*
T0*0
_output_shapes
:€€€€€€€€€А^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€А  А
flatten/ReshapeReshaperot_inv_pool/Max:output:0flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€АБ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	А *
dtype0З
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0И
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ \

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ Д
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Л
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€В
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€g
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ќ

NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp(^rot_equiv_conv2d/BiasAdd/ReadVariableOp ^rot_equiv_conv2d/ReadVariableOp"^rot_equiv_conv2d/ReadVariableOp_1"^rot_equiv_conv2d/ReadVariableOp_2,^rot_equiv_conv2d/convolution/ReadVariableOp*^rot_equiv_conv2d_1/BiasAdd/ReadVariableOp.^rot_equiv_conv2d_1/convolution/ReadVariableOp0^rot_equiv_conv2d_1/convolution_1/ReadVariableOp0^rot_equiv_conv2d_1/convolution_2/ReadVariableOp0^rot_equiv_conv2d_1/convolution_3/ReadVariableOp*^rot_equiv_conv2d_2/BiasAdd/ReadVariableOp.^rot_equiv_conv2d_2/convolution/ReadVariableOp0^rot_equiv_conv2d_2/convolution_1/ReadVariableOp0^rot_equiv_conv2d_2/convolution_2/ReadVariableOp0^rot_equiv_conv2d_2/convolution_3/ReadVariableOp*^rot_equiv_conv2d_3/BiasAdd/ReadVariableOp.^rot_equiv_conv2d_3/convolution/ReadVariableOp0^rot_equiv_conv2d_3/convolution_1/ReadVariableOp0^rot_equiv_conv2d_3/convolution_2/ReadVariableOp0^rot_equiv_conv2d_3/convolution_3/ReadVariableOp*^rot_equiv_conv2d_4/BiasAdd/ReadVariableOp.^rot_equiv_conv2d_4/convolution/ReadVariableOp0^rot_equiv_conv2d_4/convolution_1/ReadVariableOp0^rot_equiv_conv2d_4/convolution_2/ReadVariableOp0^rot_equiv_conv2d_4/convolution_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:€€€€€€€€€РР: : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2R
'rot_equiv_conv2d/BiasAdd/ReadVariableOp'rot_equiv_conv2d/BiasAdd/ReadVariableOp2B
rot_equiv_conv2d/ReadVariableOprot_equiv_conv2d/ReadVariableOp2F
!rot_equiv_conv2d/ReadVariableOp_1!rot_equiv_conv2d/ReadVariableOp_12F
!rot_equiv_conv2d/ReadVariableOp_2!rot_equiv_conv2d/ReadVariableOp_22Z
+rot_equiv_conv2d/convolution/ReadVariableOp+rot_equiv_conv2d/convolution/ReadVariableOp2V
)rot_equiv_conv2d_1/BiasAdd/ReadVariableOp)rot_equiv_conv2d_1/BiasAdd/ReadVariableOp2^
-rot_equiv_conv2d_1/convolution/ReadVariableOp-rot_equiv_conv2d_1/convolution/ReadVariableOp2b
/rot_equiv_conv2d_1/convolution_1/ReadVariableOp/rot_equiv_conv2d_1/convolution_1/ReadVariableOp2b
/rot_equiv_conv2d_1/convolution_2/ReadVariableOp/rot_equiv_conv2d_1/convolution_2/ReadVariableOp2b
/rot_equiv_conv2d_1/convolution_3/ReadVariableOp/rot_equiv_conv2d_1/convolution_3/ReadVariableOp2V
)rot_equiv_conv2d_2/BiasAdd/ReadVariableOp)rot_equiv_conv2d_2/BiasAdd/ReadVariableOp2^
-rot_equiv_conv2d_2/convolution/ReadVariableOp-rot_equiv_conv2d_2/convolution/ReadVariableOp2b
/rot_equiv_conv2d_2/convolution_1/ReadVariableOp/rot_equiv_conv2d_2/convolution_1/ReadVariableOp2b
/rot_equiv_conv2d_2/convolution_2/ReadVariableOp/rot_equiv_conv2d_2/convolution_2/ReadVariableOp2b
/rot_equiv_conv2d_2/convolution_3/ReadVariableOp/rot_equiv_conv2d_2/convolution_3/ReadVariableOp2V
)rot_equiv_conv2d_3/BiasAdd/ReadVariableOp)rot_equiv_conv2d_3/BiasAdd/ReadVariableOp2^
-rot_equiv_conv2d_3/convolution/ReadVariableOp-rot_equiv_conv2d_3/convolution/ReadVariableOp2b
/rot_equiv_conv2d_3/convolution_1/ReadVariableOp/rot_equiv_conv2d_3/convolution_1/ReadVariableOp2b
/rot_equiv_conv2d_3/convolution_2/ReadVariableOp/rot_equiv_conv2d_3/convolution_2/ReadVariableOp2b
/rot_equiv_conv2d_3/convolution_3/ReadVariableOp/rot_equiv_conv2d_3/convolution_3/ReadVariableOp2V
)rot_equiv_conv2d_4/BiasAdd/ReadVariableOp)rot_equiv_conv2d_4/BiasAdd/ReadVariableOp2^
-rot_equiv_conv2d_4/convolution/ReadVariableOp-rot_equiv_conv2d_4/convolution/ReadVariableOp2b
/rot_equiv_conv2d_4/convolution_1/ReadVariableOp/rot_equiv_conv2d_4/convolution_1/ReadVariableOp2b
/rot_equiv_conv2d_4/convolution_2/ReadVariableOp/rot_equiv_conv2d_4/convolution_2/ReadVariableOp2b
/rot_equiv_conv2d_4/convolution_3/ReadVariableOp/rot_equiv_conv2d_4/convolution_3/ReadVariableOp:Y U
1
_output_shapes
:€€€€€€€€€РР
 
_user_specified_nameinputs
Ј
J
.__inference_max_pooling2d_layer_call_fn_627934

inputs
identityЏ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_624809Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
‘
I
-__inference_rot_inv_pool_layer_call_fn_627873

inputs
identityњ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_rot_inv_pool_layer_call_and_return_conditional_losses_625502i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€А:\ X
4
_output_shapes"
 :€€€€€€€€€А
 
_user_specified_nameinputs
ґ
Ъ
+__inference_sequential_layer_call_fn_625577
rot_equiv_conv2d_input!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@$
	unknown_7:@А
	unknown_8:	А
	unknown_9:	А 

unknown_10: 

unknown_11: 

unknown_12:
identityИҐStatefulPartitionedCallН
StatefulPartitionedCallStatefulPartitionedCallrot_equiv_conv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_625546o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:€€€€€€€€€РР: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
1
_output_shapes
:€€€€€€€€€РР
0
_user_specified_namerot_equiv_conv2d_input
С
®
3__inference_rot_equiv_conv2d_3_layer_call_fn_627653

inputs!
unknown:@@
	unknown_0:@
identityИҐStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_rot_equiv_conv2d_3_layer_call_and_return_conditional_losses_625351{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€@: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:€€€€€€€€€@
 
_user_specified_nameinputs
љH
∆
L__inference_rot_equiv_conv2d_layer_call_and_return_conditional_losses_624934

inputs=
#convolution_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐReadVariableOpҐReadVariableOp_1ҐReadVariableOp_2Ґconvolution/ReadVariableOpЖ
convolution/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: *
dtype0¶
convolutionConv2Dinputs"convolution/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ОО *
paddingVALID*
strides

Rank/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: *
dtype0F
RankConst*
_output_shapes
: *
dtype0*
value	B :M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :e
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
:t
TensorScatterUpdate/indicesConst*
_output_shapes

:*
dtype0*!
valueB"       l
TensorScatterUpdate/updatesConst*
_output_shapes
:*
dtype0*
valueB"       ї
TensorScatterUpdateTensorScatterUpdaterange:output:0$TensorScatterUpdate/indices:output:0$TensorScatterUpdate/updates:output:0*
T0*
Tindices0*
_output_shapes
:z
ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: *
dtype0H
Rank_1Const*
_output_shapes
: *
dtype0*
value	B :X
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:x
	ReverseV2	ReverseV2ReadVariableOp:value:0ReverseV2/axis:output:0*
T0*&
_output_shapes
: y
	transpose	TransposeReverseV2:output:0TensorScatterUpdate:output:0*
T0*&
_output_shapes
: У
convolution_1Conv2Dinputstranspose:y:0*
T0*1
_output_shapes
:€€€€€€€€€ОО *
paddingVALID*
strides
H
Rank_2Const*
_output_shapes
: *
dtype0*
value	B :O
range_1/startConst*
_output_shapes
: *
dtype0*
value	B : O
range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :m
range_1Rangerange_1/start:output:0Rank_2:output:0range_1/delta:output:0*
_output_shapes
:v
TensorScatterUpdate_1/indicesConst*
_output_shapes

:*
dtype0*!
valueB"      n
TensorScatterUpdate_1/updatesConst*
_output_shapes
:*
dtype0*
valueB"      √
TensorScatterUpdate_1TensorScatterUpdaterange_1:output:0&TensorScatterUpdate_1/indices:output:0&TensorScatterUpdate_1/updates:output:0*
T0*
Tindices0*
_output_shapes
:М
transpose_1	Transposeconvolution_1:output:0TensorScatterUpdate_1:output:0*
T0*1
_output_shapes
:€€€€€€€€€ОО H
Rank_3Const*
_output_shapes
: *
dtype0*
value	B :Z
ReverseV2_1/axisConst*
_output_shapes
:*
dtype0*
valueB:А
ReverseV2_1	ReverseV2transpose_1:y:0ReverseV2_1/axis:output:0*
T0*1
_output_shapes
:€€€€€€€€€ОО Б
Rank_4/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: *
dtype0H
Rank_4Const*
_output_shapes
: *
dtype0*
value	B :|
ReadVariableOp_1ReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: *
dtype0H
Rank_5Const*
_output_shapes
: *
dtype0*
value	B :Z
ReverseV2_2/axisConst*
_output_shapes
:*
dtype0*
valueB: ~
ReverseV2_2	ReverseV2ReadVariableOp_1:value:0ReverseV2_2/axis:output:0*
T0*&
_output_shapes
: H
Rank_6Const*
_output_shapes
: *
dtype0*
value	B :Z
ReverseV2_3/axisConst*
_output_shapes
:*
dtype0*
valueB:z
ReverseV2_3	ReverseV2ReverseV2_2:output:0ReverseV2_3/axis:output:0*
T0*&
_output_shapes
: Ъ
convolution_2Conv2DinputsReverseV2_3:output:0*
T0*1
_output_shapes
:€€€€€€€€€ОО *
paddingVALID*
strides
H
Rank_7Const*
_output_shapes
: *
dtype0*
value	B :H
Rank_8Const*
_output_shapes
: *
dtype0*
value	B :Z
ReverseV2_4/axisConst*
_output_shapes
:*
dtype0*
valueB:З
ReverseV2_4	ReverseV2convolution_2:output:0ReverseV2_4/axis:output:0*
T0*1
_output_shapes
:€€€€€€€€€ОО H
Rank_9Const*
_output_shapes
: *
dtype0*
value	B :Z
ReverseV2_5/axisConst*
_output_shapes
:*
dtype0*
valueB:Е
ReverseV2_5	ReverseV2ReverseV2_4:output:0ReverseV2_5/axis:output:0*
T0*1
_output_shapes
:€€€€€€€€€ОО В
Rank_10/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: *
dtype0I
Rank_10Const*
_output_shapes
: *
dtype0*
value	B :O
range_2/startConst*
_output_shapes
: *
dtype0*
value	B : O
range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :n
range_2Rangerange_2/start:output:0Rank_10:output:0range_2/delta:output:0*
_output_shapes
:v
TensorScatterUpdate_2/indicesConst*
_output_shapes

:*
dtype0*!
valueB"       n
TensorScatterUpdate_2/updatesConst*
_output_shapes
:*
dtype0*
valueB"       √
TensorScatterUpdate_2TensorScatterUpdaterange_2:output:0&TensorScatterUpdate_2/indices:output:0&TensorScatterUpdate_2/updates:output:0*
T0*
Tindices0*
_output_shapes
:|
ReadVariableOp_2ReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: *
dtype0Г
transpose_2	TransposeReadVariableOp_2:value:0TensorScatterUpdate_2:output:0*
T0*&
_output_shapes
: I
Rank_11Const*
_output_shapes
: *
dtype0*
value	B :Z
ReverseV2_6/axisConst*
_output_shapes
:*
dtype0*
valueB:u
ReverseV2_6	ReverseV2transpose_2:y:0ReverseV2_6/axis:output:0*
T0*&
_output_shapes
: Ъ
convolution_3Conv2DinputsReverseV2_6:output:0*
T0*1
_output_shapes
:€€€€€€€€€ОО *
paddingVALID*
strides
I
Rank_12Const*
_output_shapes
: *
dtype0*
value	B :O
range_3/startConst*
_output_shapes
: *
dtype0*
value	B : O
range_3/deltaConst*
_output_shapes
: *
dtype0*
value	B :n
range_3Rangerange_3/start:output:0Rank_12:output:0range_3/delta:output:0*
_output_shapes
:v
TensorScatterUpdate_3/indicesConst*
_output_shapes

:*
dtype0*!
valueB"      n
TensorScatterUpdate_3/updatesConst*
_output_shapes
:*
dtype0*
valueB"      √
TensorScatterUpdate_3TensorScatterUpdaterange_3:output:0&TensorScatterUpdate_3/indices:output:0&TensorScatterUpdate_3/updates:output:0*
T0*
Tindices0*
_output_shapes
:I
Rank_13Const*
_output_shapes
: *
dtype0*
value	B :Z
ReverseV2_7/axisConst*
_output_shapes
:*
dtype0*
valueB:З
ReverseV2_7	ReverseV2convolution_3:output:0ReverseV2_7/axis:output:0*
T0*1
_output_shapes
:€€€€€€€€€ОО К
transpose_3	TransposeReverseV2_7:output:0TensorScatterUpdate_3:output:0*
T0*1
_output_shapes
:€€€€€€€€€ОО Њ
stackPackconvolution:output:0ReverseV2_1:output:0ReverseV2_5:output:0transpose_3:y:0*
N*
T0*5
_output_shapes#
!:€€€€€€€€€ОО *
axisю€€€€€€€€\
ReluRelustack:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€ОО r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ж
BiasAddBiasAddRelu:activations:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:€€€€€€€€€ОО m
IdentityIdentityBiasAdd:output:0^NoOp*
T0*5
_output_shapes#
!:€€€€€€€€€ОО ≥
NoOpNoOp^BiasAdd/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^convolution/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€РР: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_228
convolution/ReadVariableOpconvolution/ReadVariableOp:Y U
1
_output_shapes
:€€€€€€€€€РР
 
_user_specified_nameinputs
∆	
ф
C__inference_dense_1_layer_call_and_return_conditional_losses_625539

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
«
_
C__inference_flatten_layer_call_and_return_conditional_losses_625510

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€А  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€АY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
еC
о
N__inference_rot_equiv_conv2d_1_layer_call_and_return_conditional_losses_627433

inputs=
#convolution_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐconvolution/ReadVariableOpҐconvolution_1/ReadVariableOpҐconvolution_2/ReadVariableOpҐconvolution_3/ReadVariableOpG
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R K
ShapeShapeinputs*
T0*
_output_shapes
:*
out_type0	f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskG
sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RS
subSubstrided_slice:output:0sub/y:output:0*
T0	*
_output_shapes
: Z
clip_by_value/MinimumMinimumConst:output:0sub:z:0*
T0	*
_output_shapes
: Q
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R n
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*
_output_shapes
: X
GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€§
GatherV2GatherV2inputsclip_by_value:z:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€GG Ж
convolution/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:  *
dtype0ѓ
convolutionConv2DGatherV2:output:0"convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€EE *
paddingVALID*
strides
I
Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_1Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_1Substrided_slice_1:output:0sub_1/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_1/MinimumMinimumConst_1:output:0	sub_1:z:0*
T0	*
_output_shapes
: S
clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€™

GatherV2_1GatherV2inputsclip_by_value_1:z:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€GG И
convolution_1/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:  *
dtype0µ
convolution_1Conv2DGatherV2_1:output:0$convolution_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€EE *
paddingVALID*
strides
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_2Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_2StridedSliceShape_2:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_2Substrided_slice_2:output:0sub_2/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_2/MinimumMinimumConst_2:output:0	sub_2:z:0*
T0	*
_output_shapes
: S
clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€™

GatherV2_2GatherV2inputsclip_by_value_2:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€GG И
convolution_2/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:  *
dtype0µ
convolution_2Conv2DGatherV2_2:output:0$convolution_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€EE *
paddingVALID*
strides
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_3Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€j
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_3StridedSliceShape_3:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_3Substrided_slice_3:output:0sub_3/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_3/MinimumMinimumConst_3:output:0	sub_3:z:0*
T0	*
_output_shapes
: S
clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_3Maximumclip_by_value_3/Minimum:z:0clip_by_value_3/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€™

GatherV2_3GatherV2inputsclip_by_value_3:z:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€GG И
convolution_3/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:  *
dtype0µ
convolution_3Conv2DGatherV2_3:output:0$convolution_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€EE *
paddingVALID*
strides
«
stackPackconvolution:output:0convolution_1:output:0convolution_2:output:0convolution_3:output:0*
N*
T0*3
_output_shapes!
:€€€€€€€€€EE *
axisю€€€€€€€€Z
ReluRelustack:output:0*
T0*3
_output_shapes!
:€€€€€€€€€EE r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0Д
BiasAddBiasAddRelu:activations:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€EE k
IdentityIdentityBiasAdd:output:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€EE ў
NoOpNoOp^BiasAdd/ReadVariableOp^convolution/ReadVariableOp^convolution_1/ReadVariableOp^convolution_2/ReadVariableOp^convolution_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€GG : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp28
convolution/ReadVariableOpconvolution/ReadVariableOp2<
convolution_1/ReadVariableOpconvolution_1/ReadVariableOp2<
convolution_2/ReadVariableOpconvolution_2/ReadVariableOp2<
convolution_3/ReadVariableOpconvolution_3/ReadVariableOp:[ W
3
_output_shapes!
:€€€€€€€€€GG 
 
_user_specified_nameinputs
С
®
3__inference_rot_equiv_conv2d_2_layer_call_fn_627508

inputs!
unknown: @
	unknown_0:@
identityИҐStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_rot_equiv_conv2d_2_layer_call_and_return_conditional_losses_625212{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€  @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€"" : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:€€€€€€€€€"" 
 
_user_specified_nameinputs
У
g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_624821

inputs
identityҐ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
У
g
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_627969

inputs
identityҐ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
д
O
3__inference_rot_equiv_pool2d_2_layer_call_fn_627583

inputs
identity»
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_rot_equiv_pool2d_2_layer_call_and_return_conditional_losses_625279l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€  @:[ W
3
_output_shapes!
:€€€€€€€€€  @
 
_user_specified_nameinputs
К
У
$__inference_signature_wrapper_625958
rot_equiv_conv2d_input!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@$
	unknown_7:@А
	unknown_8:	А
	unknown_9:	А 

unknown_10: 

unknown_11: 

unknown_12:
identityИҐStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallrot_equiv_conv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В **
f%R#
!__inference__wrapped_model_624800o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:€€€€€€€€€РР: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
1
_output_shapes
:€€€€€€€€€РР
0
_user_specified_namerot_equiv_conv2d_input
љH
∆
L__inference_rot_equiv_conv2d_layer_call_and_return_conditional_losses_627288

inputs=
#convolution_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐReadVariableOpҐReadVariableOp_1ҐReadVariableOp_2Ґconvolution/ReadVariableOpЖ
convolution/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: *
dtype0¶
convolutionConv2Dinputs"convolution/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ОО *
paddingVALID*
strides

Rank/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: *
dtype0F
RankConst*
_output_shapes
: *
dtype0*
value	B :M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :e
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
:t
TensorScatterUpdate/indicesConst*
_output_shapes

:*
dtype0*!
valueB"       l
TensorScatterUpdate/updatesConst*
_output_shapes
:*
dtype0*
valueB"       ї
TensorScatterUpdateTensorScatterUpdaterange:output:0$TensorScatterUpdate/indices:output:0$TensorScatterUpdate/updates:output:0*
T0*
Tindices0*
_output_shapes
:z
ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: *
dtype0H
Rank_1Const*
_output_shapes
: *
dtype0*
value	B :X
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:x
	ReverseV2	ReverseV2ReadVariableOp:value:0ReverseV2/axis:output:0*
T0*&
_output_shapes
: y
	transpose	TransposeReverseV2:output:0TensorScatterUpdate:output:0*
T0*&
_output_shapes
: У
convolution_1Conv2Dinputstranspose:y:0*
T0*1
_output_shapes
:€€€€€€€€€ОО *
paddingVALID*
strides
H
Rank_2Const*
_output_shapes
: *
dtype0*
value	B :O
range_1/startConst*
_output_shapes
: *
dtype0*
value	B : O
range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :m
range_1Rangerange_1/start:output:0Rank_2:output:0range_1/delta:output:0*
_output_shapes
:v
TensorScatterUpdate_1/indicesConst*
_output_shapes

:*
dtype0*!
valueB"      n
TensorScatterUpdate_1/updatesConst*
_output_shapes
:*
dtype0*
valueB"      √
TensorScatterUpdate_1TensorScatterUpdaterange_1:output:0&TensorScatterUpdate_1/indices:output:0&TensorScatterUpdate_1/updates:output:0*
T0*
Tindices0*
_output_shapes
:М
transpose_1	Transposeconvolution_1:output:0TensorScatterUpdate_1:output:0*
T0*1
_output_shapes
:€€€€€€€€€ОО H
Rank_3Const*
_output_shapes
: *
dtype0*
value	B :Z
ReverseV2_1/axisConst*
_output_shapes
:*
dtype0*
valueB:А
ReverseV2_1	ReverseV2transpose_1:y:0ReverseV2_1/axis:output:0*
T0*1
_output_shapes
:€€€€€€€€€ОО Б
Rank_4/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: *
dtype0H
Rank_4Const*
_output_shapes
: *
dtype0*
value	B :|
ReadVariableOp_1ReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: *
dtype0H
Rank_5Const*
_output_shapes
: *
dtype0*
value	B :Z
ReverseV2_2/axisConst*
_output_shapes
:*
dtype0*
valueB: ~
ReverseV2_2	ReverseV2ReadVariableOp_1:value:0ReverseV2_2/axis:output:0*
T0*&
_output_shapes
: H
Rank_6Const*
_output_shapes
: *
dtype0*
value	B :Z
ReverseV2_3/axisConst*
_output_shapes
:*
dtype0*
valueB:z
ReverseV2_3	ReverseV2ReverseV2_2:output:0ReverseV2_3/axis:output:0*
T0*&
_output_shapes
: Ъ
convolution_2Conv2DinputsReverseV2_3:output:0*
T0*1
_output_shapes
:€€€€€€€€€ОО *
paddingVALID*
strides
H
Rank_7Const*
_output_shapes
: *
dtype0*
value	B :H
Rank_8Const*
_output_shapes
: *
dtype0*
value	B :Z
ReverseV2_4/axisConst*
_output_shapes
:*
dtype0*
valueB:З
ReverseV2_4	ReverseV2convolution_2:output:0ReverseV2_4/axis:output:0*
T0*1
_output_shapes
:€€€€€€€€€ОО H
Rank_9Const*
_output_shapes
: *
dtype0*
value	B :Z
ReverseV2_5/axisConst*
_output_shapes
:*
dtype0*
valueB:Е
ReverseV2_5	ReverseV2ReverseV2_4:output:0ReverseV2_5/axis:output:0*
T0*1
_output_shapes
:€€€€€€€€€ОО В
Rank_10/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: *
dtype0I
Rank_10Const*
_output_shapes
: *
dtype0*
value	B :O
range_2/startConst*
_output_shapes
: *
dtype0*
value	B : O
range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :n
range_2Rangerange_2/start:output:0Rank_10:output:0range_2/delta:output:0*
_output_shapes
:v
TensorScatterUpdate_2/indicesConst*
_output_shapes

:*
dtype0*!
valueB"       n
TensorScatterUpdate_2/updatesConst*
_output_shapes
:*
dtype0*
valueB"       √
TensorScatterUpdate_2TensorScatterUpdaterange_2:output:0&TensorScatterUpdate_2/indices:output:0&TensorScatterUpdate_2/updates:output:0*
T0*
Tindices0*
_output_shapes
:|
ReadVariableOp_2ReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: *
dtype0Г
transpose_2	TransposeReadVariableOp_2:value:0TensorScatterUpdate_2:output:0*
T0*&
_output_shapes
: I
Rank_11Const*
_output_shapes
: *
dtype0*
value	B :Z
ReverseV2_6/axisConst*
_output_shapes
:*
dtype0*
valueB:u
ReverseV2_6	ReverseV2transpose_2:y:0ReverseV2_6/axis:output:0*
T0*&
_output_shapes
: Ъ
convolution_3Conv2DinputsReverseV2_6:output:0*
T0*1
_output_shapes
:€€€€€€€€€ОО *
paddingVALID*
strides
I
Rank_12Const*
_output_shapes
: *
dtype0*
value	B :O
range_3/startConst*
_output_shapes
: *
dtype0*
value	B : O
range_3/deltaConst*
_output_shapes
: *
dtype0*
value	B :n
range_3Rangerange_3/start:output:0Rank_12:output:0range_3/delta:output:0*
_output_shapes
:v
TensorScatterUpdate_3/indicesConst*
_output_shapes

:*
dtype0*!
valueB"      n
TensorScatterUpdate_3/updatesConst*
_output_shapes
:*
dtype0*
valueB"      √
TensorScatterUpdate_3TensorScatterUpdaterange_3:output:0&TensorScatterUpdate_3/indices:output:0&TensorScatterUpdate_3/updates:output:0*
T0*
Tindices0*
_output_shapes
:I
Rank_13Const*
_output_shapes
: *
dtype0*
value	B :Z
ReverseV2_7/axisConst*
_output_shapes
:*
dtype0*
valueB:З
ReverseV2_7	ReverseV2convolution_3:output:0ReverseV2_7/axis:output:0*
T0*1
_output_shapes
:€€€€€€€€€ОО К
transpose_3	TransposeReverseV2_7:output:0TensorScatterUpdate_3:output:0*
T0*1
_output_shapes
:€€€€€€€€€ОО Њ
stackPackconvolution:output:0ReverseV2_1:output:0ReverseV2_5:output:0transpose_3:y:0*
N*
T0*5
_output_shapes#
!:€€€€€€€€€ОО *
axisю€€€€€€€€\
ReluRelustack:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€ОО r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ж
BiasAddBiasAddRelu:activations:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:€€€€€€€€€ОО m
IdentityIdentityBiasAdd:output:0^NoOp*
T0*5
_output_shapes#
!:€€€€€€€€€ОО ≥
NoOpNoOp^BiasAdd/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^convolution/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€РР: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_228
convolution/ReadVariableOpconvolution/ReadVariableOp:Y U
1
_output_shapes
:€€€€€€€€€РР
 
_user_specified_nameinputs
ї
L
0__inference_max_pooling2d_1_layer_call_fn_627944

inputs
identity№
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_624821Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
∆6
j
N__inference_rot_equiv_pool2d_3_layer_call_and_return_conditional_losses_627789

inputs
identityG
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R K
ShapeShapeinputs*
T0*
_output_shapes
:*
out_type0	f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskG
sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RS
subSubstrided_slice:output:0sub/y:output:0*
T0	*
_output_shapes
: Z
clip_by_value/MinimumMinimumConst:output:0sub:z:0*
T0	*
_output_shapes
: Q
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R n
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*
_output_shapes
: X
GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€§
GatherV2GatherV2inputsclip_by_value:z:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@Ґ
max_pooling2d_3/MaxPoolMaxPoolGatherV2:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
I
Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_1Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_1Substrided_slice_1:output:0sub_1/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_1/MinimumMinimumConst_1:output:0	sub_1:z:0*
T0	*
_output_shapes
: S
clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€™

GatherV2_1GatherV2inputsclip_by_value_1:z:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@¶
max_pooling2d_3/MaxPool_1MaxPoolGatherV2_1:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_2Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_2StridedSliceShape_2:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_2Substrided_slice_2:output:0sub_2/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_2/MinimumMinimumConst_2:output:0	sub_2:z:0*
T0	*
_output_shapes
: S
clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€™

GatherV2_2GatherV2inputsclip_by_value_2:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@¶
max_pooling2d_3/MaxPool_2MaxPoolGatherV2_2:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_3Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€j
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_3StridedSliceShape_3:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_3Substrided_slice_3:output:0sub_3/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_3/MinimumMinimumConst_3:output:0	sub_3:z:0*
T0	*
_output_shapes
: S
clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_3Maximumclip_by_value_3/Minimum:z:0clip_by_value_3/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€™

GatherV2_3GatherV2inputsclip_by_value_3:z:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@¶
max_pooling2d_3/MaxPool_3MaxPoolGatherV2_3:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
ч
stackPack max_pooling2d_3/MaxPool:output:0"max_pooling2d_3/MaxPool_1:output:0"max_pooling2d_3/MaxPool_2:output:0"max_pooling2d_3/MaxPool_3:output:0*
N*
T0*3
_output_shapes!
:€€€€€€€€€@*
axisю€€€€€€€€b
IdentityIdentitystack:output:0*
T0*3
_output_shapes!
:€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€@:[ W
3
_output_shapes!
:€€€€€€€€€@
 
_user_specified_nameinputs
д;
„
F__inference_sequential_layer_call_and_return_conditional_losses_625546

inputs1
rot_equiv_conv2d_624935: %
rot_equiv_conv2d_624937: 3
rot_equiv_conv2d_1_625074:  '
rot_equiv_conv2d_1_625076: 3
rot_equiv_conv2d_2_625213: @'
rot_equiv_conv2d_2_625215:@3
rot_equiv_conv2d_3_625352:@@'
rot_equiv_conv2d_3_625354:@4
rot_equiv_conv2d_4_625491:@А(
rot_equiv_conv2d_4_625493:	А
dense_625524:	А 
dense_625526:  
dense_1_625540: 
dense_1_625542:
identityИҐdense/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐ(rot_equiv_conv2d/StatefulPartitionedCallҐ*rot_equiv_conv2d_1/StatefulPartitionedCallҐ*rot_equiv_conv2d_2/StatefulPartitionedCallҐ*rot_equiv_conv2d_3/StatefulPartitionedCallҐ*rot_equiv_conv2d_4/StatefulPartitionedCall°
(rot_equiv_conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsrot_equiv_conv2d_624935rot_equiv_conv2d_624937*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:€€€€€€€€€ОО *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_rot_equiv_conv2d_layer_call_and_return_conditional_losses_624934В
 rot_equiv_pool2d/PartitionedCallPartitionedCall1rot_equiv_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€GG * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_rot_equiv_pool2d_layer_call_and_return_conditional_losses_625001 
*rot_equiv_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall)rot_equiv_pool2d/PartitionedCall:output:0rot_equiv_conv2d_1_625074rot_equiv_conv2d_1_625076*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€EE *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_rot_equiv_conv2d_1_layer_call_and_return_conditional_losses_625073И
"rot_equiv_pool2d_1/PartitionedCallPartitionedCall3rot_equiv_conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€"" * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_rot_equiv_pool2d_1_layer_call_and_return_conditional_losses_625140ћ
*rot_equiv_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall+rot_equiv_pool2d_1/PartitionedCall:output:0rot_equiv_conv2d_2_625213rot_equiv_conv2d_2_625215*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_rot_equiv_conv2d_2_layer_call_and_return_conditional_losses_625212И
"rot_equiv_pool2d_2/PartitionedCallPartitionedCall3rot_equiv_conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_rot_equiv_pool2d_2_layer_call_and_return_conditional_losses_625279ћ
*rot_equiv_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall+rot_equiv_pool2d_2/PartitionedCall:output:0rot_equiv_conv2d_3_625352rot_equiv_conv2d_3_625354*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_rot_equiv_conv2d_3_layer_call_and_return_conditional_losses_625351И
"rot_equiv_pool2d_3/PartitionedCallPartitionedCall3rot_equiv_conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_rot_equiv_pool2d_3_layer_call_and_return_conditional_losses_625418Ќ
*rot_equiv_conv2d_4/StatefulPartitionedCallStatefulPartitionedCall+rot_equiv_pool2d_3/PartitionedCall:output:0rot_equiv_conv2d_4_625491rot_equiv_conv2d_4_625493*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_rot_equiv_conv2d_4_layer_call_and_return_conditional_losses_625490щ
rot_inv_pool/PartitionedCallPartitionedCall3rot_equiv_conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_rot_inv_pool_layer_call_and_return_conditional_losses_625502ў
flatten/PartitionedCallPartitionedCall%rot_inv_pool/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_625510Б
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_625524dense_625526*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_625523П
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_625540dense_1_625542*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_625539w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€з
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall)^rot_equiv_conv2d/StatefulPartitionedCall+^rot_equiv_conv2d_1/StatefulPartitionedCall+^rot_equiv_conv2d_2/StatefulPartitionedCall+^rot_equiv_conv2d_3/StatefulPartitionedCall+^rot_equiv_conv2d_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:€€€€€€€€€РР: : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2T
(rot_equiv_conv2d/StatefulPartitionedCall(rot_equiv_conv2d/StatefulPartitionedCall2X
*rot_equiv_conv2d_1/StatefulPartitionedCall*rot_equiv_conv2d_1/StatefulPartitionedCall2X
*rot_equiv_conv2d_2/StatefulPartitionedCall*rot_equiv_conv2d_2/StatefulPartitionedCall2X
*rot_equiv_conv2d_3/StatefulPartitionedCall*rot_equiv_conv2d_3/StatefulPartitionedCall2X
*rot_equiv_conv2d_4/StatefulPartitionedCall*rot_equiv_conv2d_4/StatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€РР
 
_user_specified_nameinputs
еC
о
N__inference_rot_equiv_conv2d_2_layer_call_and_return_conditional_losses_625212

inputs=
#convolution_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐconvolution/ReadVariableOpҐconvolution_1/ReadVariableOpҐconvolution_2/ReadVariableOpҐconvolution_3/ReadVariableOpG
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R K
ShapeShapeinputs*
T0*
_output_shapes
:*
out_type0	f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskG
sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RS
subSubstrided_slice:output:0sub/y:output:0*
T0	*
_output_shapes
: Z
clip_by_value/MinimumMinimumConst:output:0sub:z:0*
T0	*
_output_shapes
: Q
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R n
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*
_output_shapes
: X
GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€§
GatherV2GatherV2inputsclip_by_value:z:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€"" Ж
convolution/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0ѓ
convolutionConv2DGatherV2:output:0"convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  @*
paddingVALID*
strides
I
Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_1Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_1Substrided_slice_1:output:0sub_1/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_1/MinimumMinimumConst_1:output:0	sub_1:z:0*
T0	*
_output_shapes
: S
clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€™

GatherV2_1GatherV2inputsclip_by_value_1:z:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€"" И
convolution_1/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0µ
convolution_1Conv2DGatherV2_1:output:0$convolution_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  @*
paddingVALID*
strides
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_2Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_2StridedSliceShape_2:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_2Substrided_slice_2:output:0sub_2/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_2/MinimumMinimumConst_2:output:0	sub_2:z:0*
T0	*
_output_shapes
: S
clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€™

GatherV2_2GatherV2inputsclip_by_value_2:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€"" И
convolution_2/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0µ
convolution_2Conv2DGatherV2_2:output:0$convolution_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  @*
paddingVALID*
strides
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_3Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€j
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_3StridedSliceShape_3:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_3Substrided_slice_3:output:0sub_3/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_3/MinimumMinimumConst_3:output:0	sub_3:z:0*
T0	*
_output_shapes
: S
clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_3Maximumclip_by_value_3/Minimum:z:0clip_by_value_3/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€™

GatherV2_3GatherV2inputsclip_by_value_3:z:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€"" И
convolution_3/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0µ
convolution_3Conv2DGatherV2_3:output:0$convolution_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  @*
paddingVALID*
strides
«
stackPackconvolution:output:0convolution_1:output:0convolution_2:output:0convolution_3:output:0*
N*
T0*3
_output_shapes!
:€€€€€€€€€  @*
axisю€€€€€€€€Z
ReluRelustack:output:0*
T0*3
_output_shapes!
:€€€€€€€€€  @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Д
BiasAddBiasAddRelu:activations:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€  @k
IdentityIdentityBiasAdd:output:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€  @ў
NoOpNoOp^BiasAdd/ReadVariableOp^convolution/ReadVariableOp^convolution_1/ReadVariableOp^convolution_2/ReadVariableOp^convolution_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€"" : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp28
convolution/ReadVariableOpconvolution/ReadVariableOp2<
convolution_1/ReadVariableOpconvolution_1/ReadVariableOp2<
convolution_2/ReadVariableOpconvolution_2/ReadVariableOp2<
convolution_3/ReadVariableOpconvolution_3/ReadVariableOp:[ W
3
_output_shapes!
:€€€€€€€€€"" 
 
_user_specified_nameinputs
у”
∆"
"__inference__traced_restore_628314
file_prefixB
(assignvariableop_rot_equiv_conv2d_kernel: 6
(assignvariableop_1_rot_equiv_conv2d_bias: F
,assignvariableop_2_rot_equiv_conv2d_1_kernel:  8
*assignvariableop_3_rot_equiv_conv2d_1_bias: F
,assignvariableop_4_rot_equiv_conv2d_2_kernel: @8
*assignvariableop_5_rot_equiv_conv2d_2_bias:@F
,assignvariableop_6_rot_equiv_conv2d_3_kernel:@@8
*assignvariableop_7_rot_equiv_conv2d_3_bias:@G
,assignvariableop_8_rot_equiv_conv2d_4_kernel:@А9
*assignvariableop_9_rot_equiv_conv2d_4_bias:	А3
 assignvariableop_10_dense_kernel:	А ,
assignvariableop_11_dense_bias: 4
"assignvariableop_12_dense_1_kernel: .
 assignvariableop_13_dense_1_bias:(
assignvariableop_14_nadam_iter:	 *
 assignvariableop_15_nadam_beta_1: *
 assignvariableop_16_nadam_beta_2: )
assignvariableop_17_nadam_decay: 1
'assignvariableop_18_nadam_learning_rate: 2
(assignvariableop_19_nadam_momentum_cache: %
assignvariableop_20_total_1: %
assignvariableop_21_count_1: #
assignvariableop_22_total: #
assignvariableop_23_count: M
3assignvariableop_24_nadam_rot_equiv_conv2d_kernel_m: ?
1assignvariableop_25_nadam_rot_equiv_conv2d_bias_m: O
5assignvariableop_26_nadam_rot_equiv_conv2d_1_kernel_m:  A
3assignvariableop_27_nadam_rot_equiv_conv2d_1_bias_m: O
5assignvariableop_28_nadam_rot_equiv_conv2d_2_kernel_m: @A
3assignvariableop_29_nadam_rot_equiv_conv2d_2_bias_m:@O
5assignvariableop_30_nadam_rot_equiv_conv2d_3_kernel_m:@@A
3assignvariableop_31_nadam_rot_equiv_conv2d_3_bias_m:@P
5assignvariableop_32_nadam_rot_equiv_conv2d_4_kernel_m:@АB
3assignvariableop_33_nadam_rot_equiv_conv2d_4_bias_m:	А;
(assignvariableop_34_nadam_dense_kernel_m:	А 4
&assignvariableop_35_nadam_dense_bias_m: <
*assignvariableop_36_nadam_dense_1_kernel_m: 6
(assignvariableop_37_nadam_dense_1_bias_m:M
3assignvariableop_38_nadam_rot_equiv_conv2d_kernel_v: ?
1assignvariableop_39_nadam_rot_equiv_conv2d_bias_v: O
5assignvariableop_40_nadam_rot_equiv_conv2d_1_kernel_v:  A
3assignvariableop_41_nadam_rot_equiv_conv2d_1_bias_v: O
5assignvariableop_42_nadam_rot_equiv_conv2d_2_kernel_v: @A
3assignvariableop_43_nadam_rot_equiv_conv2d_2_bias_v:@O
5assignvariableop_44_nadam_rot_equiv_conv2d_3_kernel_v:@@A
3assignvariableop_45_nadam_rot_equiv_conv2d_3_bias_v:@P
5assignvariableop_46_nadam_rot_equiv_conv2d_4_kernel_v:@АB
3assignvariableop_47_nadam_rot_equiv_conv2d_4_bias_v:	А;
(assignvariableop_48_nadam_dense_kernel_v:	А 4
&assignvariableop_49_nadam_dense_bias_v: <
*assignvariableop_50_nadam_dense_1_kernel_v: 6
(assignvariableop_51_nadam_dense_1_bias_v:
identity_53ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_33ҐAssignVariableOp_34ҐAssignVariableOp_35ҐAssignVariableOp_36ҐAssignVariableOp_37ҐAssignVariableOp_38ҐAssignVariableOp_39ҐAssignVariableOp_4ҐAssignVariableOp_40ҐAssignVariableOp_41ҐAssignVariableOp_42ҐAssignVariableOp_43ҐAssignVariableOp_44ҐAssignVariableOp_45ҐAssignVariableOp_46ҐAssignVariableOp_47ҐAssignVariableOp_48ҐAssignVariableOp_49ҐAssignVariableOp_5ҐAssignVariableOp_50ҐAssignVariableOp_51ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9Х
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:5*
dtype0*ї
value±BЃ5B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЏ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:5*
dtype0*}
valuetBr5B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ™
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*к
_output_shapes„
‘:::::::::::::::::::::::::::::::::::::::::::::::::::::*C
dtypes9
725	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOpAssignVariableOp(assignvariableop_rot_equiv_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_1AssignVariableOp(assignvariableop_1_rot_equiv_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_2AssignVariableOp,assignvariableop_2_rot_equiv_conv2d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_3AssignVariableOp*assignvariableop_3_rot_equiv_conv2d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_4AssignVariableOp,assignvariableop_4_rot_equiv_conv2d_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_5AssignVariableOp*assignvariableop_5_rot_equiv_conv2d_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_6AssignVariableOp,assignvariableop_6_rot_equiv_conv2d_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_7AssignVariableOp*assignvariableop_7_rot_equiv_conv2d_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_8AssignVariableOp,assignvariableop_8_rot_equiv_conv2d_4_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_9AssignVariableOp*assignvariableop_9_rot_equiv_conv2d_4_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_10AssignVariableOp assignvariableop_10_dense_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_11AssignVariableOpassignvariableop_11_dense_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_1_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_1_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0	*
_output_shapes
:П
AssignVariableOp_14AssignVariableOpassignvariableop_14_nadam_iterIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_15AssignVariableOp assignvariableop_15_nadam_beta_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_16AssignVariableOp assignvariableop_16_nadam_beta_2Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_17AssignVariableOpassignvariableop_17_nadam_decayIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_18AssignVariableOp'assignvariableop_18_nadam_learning_rateIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_19AssignVariableOp(assignvariableop_19_nadam_momentum_cacheIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_20AssignVariableOpassignvariableop_20_total_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_21AssignVariableOpassignvariableop_21_count_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_22AssignVariableOpassignvariableop_22_totalIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_23AssignVariableOpassignvariableop_23_countIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_24AssignVariableOp3assignvariableop_24_nadam_rot_equiv_conv2d_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Ґ
AssignVariableOp_25AssignVariableOp1assignvariableop_25_nadam_rot_equiv_conv2d_bias_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_26AssignVariableOp5assignvariableop_26_nadam_rot_equiv_conv2d_1_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_27AssignVariableOp3assignvariableop_27_nadam_rot_equiv_conv2d_1_bias_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_28AssignVariableOp5assignvariableop_28_nadam_rot_equiv_conv2d_2_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_29AssignVariableOp3assignvariableop_29_nadam_rot_equiv_conv2d_2_bias_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_30AssignVariableOp5assignvariableop_30_nadam_rot_equiv_conv2d_3_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_31AssignVariableOp3assignvariableop_31_nadam_rot_equiv_conv2d_3_bias_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_32AssignVariableOp5assignvariableop_32_nadam_rot_equiv_conv2d_4_kernel_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_33AssignVariableOp3assignvariableop_33_nadam_rot_equiv_conv2d_4_bias_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_34AssignVariableOp(assignvariableop_34_nadam_dense_kernel_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_35AssignVariableOp&assignvariableop_35_nadam_dense_bias_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_36AssignVariableOp*assignvariableop_36_nadam_dense_1_kernel_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_37AssignVariableOp(assignvariableop_37_nadam_dense_1_bias_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_38AssignVariableOp3assignvariableop_38_nadam_rot_equiv_conv2d_kernel_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:Ґ
AssignVariableOp_39AssignVariableOp1assignvariableop_39_nadam_rot_equiv_conv2d_bias_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_40AssignVariableOp5assignvariableop_40_nadam_rot_equiv_conv2d_1_kernel_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_41AssignVariableOp3assignvariableop_41_nadam_rot_equiv_conv2d_1_bias_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_42AssignVariableOp5assignvariableop_42_nadam_rot_equiv_conv2d_2_kernel_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_43AssignVariableOp3assignvariableop_43_nadam_rot_equiv_conv2d_2_bias_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_44AssignVariableOp5assignvariableop_44_nadam_rot_equiv_conv2d_3_kernel_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_45AssignVariableOp3assignvariableop_45_nadam_rot_equiv_conv2d_3_bias_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_46AssignVariableOp5assignvariableop_46_nadam_rot_equiv_conv2d_4_kernel_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_47AssignVariableOp3assignvariableop_47_nadam_rot_equiv_conv2d_4_bias_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_48AssignVariableOp(assignvariableop_48_nadam_dense_kernel_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_49AssignVariableOp&assignvariableop_49_nadam_dense_bias_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_50AssignVariableOp*assignvariableop_50_nadam_dense_1_kernel_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_51AssignVariableOp(assignvariableop_51_nadam_dense_1_bias_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 «	
Identity_52Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_53IdentityIdentity_52:output:0^NoOp_1*
T0*
_output_shapes
: і	
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_53Identity_53:output:0*}
_input_shapesl
j: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ф<
з
F__inference_sequential_layer_call_and_return_conditional_losses_625917
rot_equiv_conv2d_input1
rot_equiv_conv2d_625875: %
rot_equiv_conv2d_625877: 3
rot_equiv_conv2d_1_625881:  '
rot_equiv_conv2d_1_625883: 3
rot_equiv_conv2d_2_625887: @'
rot_equiv_conv2d_2_625889:@3
rot_equiv_conv2d_3_625893:@@'
rot_equiv_conv2d_3_625895:@4
rot_equiv_conv2d_4_625899:@А(
rot_equiv_conv2d_4_625901:	А
dense_625906:	А 
dense_625908:  
dense_1_625911: 
dense_1_625913:
identityИҐdense/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐ(rot_equiv_conv2d/StatefulPartitionedCallҐ*rot_equiv_conv2d_1/StatefulPartitionedCallҐ*rot_equiv_conv2d_2/StatefulPartitionedCallҐ*rot_equiv_conv2d_3/StatefulPartitionedCallҐ*rot_equiv_conv2d_4/StatefulPartitionedCall±
(rot_equiv_conv2d/StatefulPartitionedCallStatefulPartitionedCallrot_equiv_conv2d_inputrot_equiv_conv2d_625875rot_equiv_conv2d_625877*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:€€€€€€€€€ОО *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_rot_equiv_conv2d_layer_call_and_return_conditional_losses_624934В
 rot_equiv_pool2d/PartitionedCallPartitionedCall1rot_equiv_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€GG * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_rot_equiv_pool2d_layer_call_and_return_conditional_losses_625001 
*rot_equiv_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall)rot_equiv_pool2d/PartitionedCall:output:0rot_equiv_conv2d_1_625881rot_equiv_conv2d_1_625883*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€EE *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_rot_equiv_conv2d_1_layer_call_and_return_conditional_losses_625073И
"rot_equiv_pool2d_1/PartitionedCallPartitionedCall3rot_equiv_conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€"" * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_rot_equiv_pool2d_1_layer_call_and_return_conditional_losses_625140ћ
*rot_equiv_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall+rot_equiv_pool2d_1/PartitionedCall:output:0rot_equiv_conv2d_2_625887rot_equiv_conv2d_2_625889*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_rot_equiv_conv2d_2_layer_call_and_return_conditional_losses_625212И
"rot_equiv_pool2d_2/PartitionedCallPartitionedCall3rot_equiv_conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_rot_equiv_pool2d_2_layer_call_and_return_conditional_losses_625279ћ
*rot_equiv_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall+rot_equiv_pool2d_2/PartitionedCall:output:0rot_equiv_conv2d_3_625893rot_equiv_conv2d_3_625895*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_rot_equiv_conv2d_3_layer_call_and_return_conditional_losses_625351И
"rot_equiv_pool2d_3/PartitionedCallPartitionedCall3rot_equiv_conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_rot_equiv_pool2d_3_layer_call_and_return_conditional_losses_625418Ќ
*rot_equiv_conv2d_4/StatefulPartitionedCallStatefulPartitionedCall+rot_equiv_pool2d_3/PartitionedCall:output:0rot_equiv_conv2d_4_625899rot_equiv_conv2d_4_625901*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_rot_equiv_conv2d_4_layer_call_and_return_conditional_losses_625490щ
rot_inv_pool/PartitionedCallPartitionedCall3rot_equiv_conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_rot_inv_pool_layer_call_and_return_conditional_losses_625502ў
flatten/PartitionedCallPartitionedCall%rot_inv_pool/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_625510Б
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_625906dense_625908*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_625523П
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_625911dense_1_625913*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_625539w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€з
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall)^rot_equiv_conv2d/StatefulPartitionedCall+^rot_equiv_conv2d_1/StatefulPartitionedCall+^rot_equiv_conv2d_2/StatefulPartitionedCall+^rot_equiv_conv2d_3/StatefulPartitionedCall+^rot_equiv_conv2d_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:€€€€€€€€€РР: : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2T
(rot_equiv_conv2d/StatefulPartitionedCall(rot_equiv_conv2d/StatefulPartitionedCall2X
*rot_equiv_conv2d_1/StatefulPartitionedCall*rot_equiv_conv2d_1/StatefulPartitionedCall2X
*rot_equiv_conv2d_2/StatefulPartitionedCall*rot_equiv_conv2d_2/StatefulPartitionedCall2X
*rot_equiv_conv2d_3/StatefulPartitionedCall*rot_equiv_conv2d_3/StatefulPartitionedCall2X
*rot_equiv_conv2d_4/StatefulPartitionedCall*rot_equiv_conv2d_4/StatefulPartitionedCall:i e
1
_output_shapes
:€€€€€€€€€РР
0
_user_specified_namerot_equiv_conv2d_input
ґ
Ъ
+__inference_sequential_layer_call_fn_625827
rot_equiv_conv2d_input!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@$
	unknown_7:@А
	unknown_8:	А
	unknown_9:	А 

unknown_10: 

unknown_11: 

unknown_12:
identityИҐStatefulPartitionedCallН
StatefulPartitionedCallStatefulPartitionedCallrot_equiv_conv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_625763o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:€€€€€€€€€РР: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
1
_output_shapes
:€€€€€€€€€РР
0
_user_specified_namerot_equiv_conv2d_input
У
g
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_624845

inputs
identityҐ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ф<
з
F__inference_sequential_layer_call_and_return_conditional_losses_625872
rot_equiv_conv2d_input1
rot_equiv_conv2d_625830: %
rot_equiv_conv2d_625832: 3
rot_equiv_conv2d_1_625836:  '
rot_equiv_conv2d_1_625838: 3
rot_equiv_conv2d_2_625842: @'
rot_equiv_conv2d_2_625844:@3
rot_equiv_conv2d_3_625848:@@'
rot_equiv_conv2d_3_625850:@4
rot_equiv_conv2d_4_625854:@А(
rot_equiv_conv2d_4_625856:	А
dense_625861:	А 
dense_625863:  
dense_1_625866: 
dense_1_625868:
identityИҐdense/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐ(rot_equiv_conv2d/StatefulPartitionedCallҐ*rot_equiv_conv2d_1/StatefulPartitionedCallҐ*rot_equiv_conv2d_2/StatefulPartitionedCallҐ*rot_equiv_conv2d_3/StatefulPartitionedCallҐ*rot_equiv_conv2d_4/StatefulPartitionedCall±
(rot_equiv_conv2d/StatefulPartitionedCallStatefulPartitionedCallrot_equiv_conv2d_inputrot_equiv_conv2d_625830rot_equiv_conv2d_625832*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:€€€€€€€€€ОО *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_rot_equiv_conv2d_layer_call_and_return_conditional_losses_624934В
 rot_equiv_pool2d/PartitionedCallPartitionedCall1rot_equiv_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€GG * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_rot_equiv_pool2d_layer_call_and_return_conditional_losses_625001 
*rot_equiv_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall)rot_equiv_pool2d/PartitionedCall:output:0rot_equiv_conv2d_1_625836rot_equiv_conv2d_1_625838*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€EE *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_rot_equiv_conv2d_1_layer_call_and_return_conditional_losses_625073И
"rot_equiv_pool2d_1/PartitionedCallPartitionedCall3rot_equiv_conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€"" * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_rot_equiv_pool2d_1_layer_call_and_return_conditional_losses_625140ћ
*rot_equiv_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall+rot_equiv_pool2d_1/PartitionedCall:output:0rot_equiv_conv2d_2_625842rot_equiv_conv2d_2_625844*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_rot_equiv_conv2d_2_layer_call_and_return_conditional_losses_625212И
"rot_equiv_pool2d_2/PartitionedCallPartitionedCall3rot_equiv_conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_rot_equiv_pool2d_2_layer_call_and_return_conditional_losses_625279ћ
*rot_equiv_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall+rot_equiv_pool2d_2/PartitionedCall:output:0rot_equiv_conv2d_3_625848rot_equiv_conv2d_3_625850*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_rot_equiv_conv2d_3_layer_call_and_return_conditional_losses_625351И
"rot_equiv_pool2d_3/PartitionedCallPartitionedCall3rot_equiv_conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_rot_equiv_pool2d_3_layer_call_and_return_conditional_losses_625418Ќ
*rot_equiv_conv2d_4/StatefulPartitionedCallStatefulPartitionedCall+rot_equiv_pool2d_3/PartitionedCall:output:0rot_equiv_conv2d_4_625854rot_equiv_conv2d_4_625856*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_rot_equiv_conv2d_4_layer_call_and_return_conditional_losses_625490щ
rot_inv_pool/PartitionedCallPartitionedCall3rot_equiv_conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_rot_inv_pool_layer_call_and_return_conditional_losses_625502ў
flatten/PartitionedCallPartitionedCall%rot_inv_pool/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_625510Б
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_625861dense_625863*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_625523П
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_625866dense_1_625868*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_625539w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€з
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall)^rot_equiv_conv2d/StatefulPartitionedCall+^rot_equiv_conv2d_1/StatefulPartitionedCall+^rot_equiv_conv2d_2/StatefulPartitionedCall+^rot_equiv_conv2d_3/StatefulPartitionedCall+^rot_equiv_conv2d_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:€€€€€€€€€РР: : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2T
(rot_equiv_conv2d/StatefulPartitionedCall(rot_equiv_conv2d/StatefulPartitionedCall2X
*rot_equiv_conv2d_1/StatefulPartitionedCall*rot_equiv_conv2d_1/StatefulPartitionedCall2X
*rot_equiv_conv2d_2/StatefulPartitionedCall*rot_equiv_conv2d_2/StatefulPartitionedCall2X
*rot_equiv_conv2d_3/StatefulPartitionedCall*rot_equiv_conv2d_3/StatefulPartitionedCall2X
*rot_equiv_conv2d_4/StatefulPartitionedCall*rot_equiv_conv2d_4/StatefulPartitionedCall:i e
1
_output_shapes
:€€€€€€€€€РР
0
_user_specified_namerot_equiv_conv2d_input
Х
™
3__inference_rot_equiv_conv2d_4_layer_call_fn_627798

inputs"
unknown:@А
	unknown_0:	А
identityИҐStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_rot_equiv_conv2d_4_layer_call_and_return_conditional_losses_625490|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€@: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:€€€€€€€€€@
 
_user_specified_nameinputs
Ж
К
+__inference_sequential_layer_call_fn_625991

inputs!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@$
	unknown_7:@А
	unknown_8:	А
	unknown_9:	А 

unknown_10: 

unknown_11: 

unknown_12:
identityИҐStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_625546o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:€€€€€€€€€РР: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€РР
 
_user_specified_nameinputs
фC
р
N__inference_rot_equiv_conv2d_4_layer_call_and_return_conditional_losses_625490

inputs>
#convolution_readvariableop_resource:@А.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐconvolution/ReadVariableOpҐconvolution_1/ReadVariableOpҐconvolution_2/ReadVariableOpҐconvolution_3/ReadVariableOpG
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R K
ShapeShapeinputs*
T0*
_output_shapes
:*
out_type0	f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskG
sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RS
subSubstrided_slice:output:0sub/y:output:0*
T0	*
_output_shapes
: Z
clip_by_value/MinimumMinimumConst:output:0sub:z:0*
T0	*
_output_shapes
: Q
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R n
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*
_output_shapes
: X
GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€§
GatherV2GatherV2inputsclip_by_value:z:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@З
convolution/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*'
_output_shapes
:@А*
dtype0∞
convolutionConv2DGatherV2:output:0"convolution/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingVALID*
strides
I
Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_1Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_1Substrided_slice_1:output:0sub_1/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_1/MinimumMinimumConst_1:output:0	sub_1:z:0*
T0	*
_output_shapes
: S
clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€™

GatherV2_1GatherV2inputsclip_by_value_1:z:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@Й
convolution_1/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*'
_output_shapes
:@А*
dtype0ґ
convolution_1Conv2DGatherV2_1:output:0$convolution_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingVALID*
strides
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_2Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_2StridedSliceShape_2:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_2Substrided_slice_2:output:0sub_2/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_2/MinimumMinimumConst_2:output:0	sub_2:z:0*
T0	*
_output_shapes
: S
clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€™

GatherV2_2GatherV2inputsclip_by_value_2:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@Й
convolution_2/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*'
_output_shapes
:@А*
dtype0ґ
convolution_2Conv2DGatherV2_2:output:0$convolution_2/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingVALID*
strides
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_3Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€j
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_3StridedSliceShape_3:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_3Substrided_slice_3:output:0sub_3/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_3/MinimumMinimumConst_3:output:0	sub_3:z:0*
T0	*
_output_shapes
: S
clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_3Maximumclip_by_value_3/Minimum:z:0clip_by_value_3/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€™

GatherV2_3GatherV2inputsclip_by_value_3:z:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@Й
convolution_3/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*'
_output_shapes
:@А*
dtype0ґ
convolution_3Conv2DGatherV2_3:output:0$convolution_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingVALID*
strides
»
stackPackconvolution:output:0convolution_1:output:0convolution_2:output:0convolution_3:output:0*
N*
T0*4
_output_shapes"
 :€€€€€€€€€А*
axisю€€€€€€€€[
ReluRelustack:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Е
BiasAddBiasAddRelu:activations:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аl
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€Аў
NoOpNoOp^BiasAdd/ReadVariableOp^convolution/ReadVariableOp^convolution_1/ReadVariableOp^convolution_2/ReadVariableOp^convolution_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp28
convolution/ReadVariableOpconvolution/ReadVariableOp2<
convolution_1/ReadVariableOpconvolution_1/ReadVariableOp2<
convolution_2/ReadVariableOpconvolution_2/ReadVariableOp2<
convolution_3/ReadVariableOpconvolution_3/ReadVariableOp:[ W
3
_output_shapes!
:€€€€€€€€€@
 
_user_specified_nameinputs
д
M
1__inference_rot_equiv_pool2d_layer_call_fn_627293

inputs
identity∆
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€GG * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_rot_equiv_pool2d_layer_call_and_return_conditional_losses_625001l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:€€€€€€€€€GG "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€ОО :] Y
5
_output_shapes#
!:€€€€€€€€€ОО 
 
_user_specified_nameinputs
∆	
ф
C__inference_dense_1_layer_call_and_return_conditional_losses_627929

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
Н
¶
1__inference_rot_equiv_conv2d_layer_call_fn_627209

inputs!
unknown: 
	unknown_0: 
identityИҐStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:€€€€€€€€€ОО *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_rot_equiv_conv2d_layer_call_and_return_conditional_losses_624934}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:€€€€€€€€€ОО `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€РР: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€РР
 
_user_specified_nameinputs
Мm
ј
__inference__traced_save_628148
file_prefix6
2savev2_rot_equiv_conv2d_kernel_read_readvariableop4
0savev2_rot_equiv_conv2d_bias_read_readvariableop8
4savev2_rot_equiv_conv2d_1_kernel_read_readvariableop6
2savev2_rot_equiv_conv2d_1_bias_read_readvariableop8
4savev2_rot_equiv_conv2d_2_kernel_read_readvariableop6
2savev2_rot_equiv_conv2d_2_bias_read_readvariableop8
4savev2_rot_equiv_conv2d_3_kernel_read_readvariableop6
2savev2_rot_equiv_conv2d_3_bias_read_readvariableop8
4savev2_rot_equiv_conv2d_4_kernel_read_readvariableop6
2savev2_rot_equiv_conv2d_4_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop)
%savev2_nadam_iter_read_readvariableop	+
'savev2_nadam_beta_1_read_readvariableop+
'savev2_nadam_beta_2_read_readvariableop*
&savev2_nadam_decay_read_readvariableop2
.savev2_nadam_learning_rate_read_readvariableop3
/savev2_nadam_momentum_cache_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop>
:savev2_nadam_rot_equiv_conv2d_kernel_m_read_readvariableop<
8savev2_nadam_rot_equiv_conv2d_bias_m_read_readvariableop@
<savev2_nadam_rot_equiv_conv2d_1_kernel_m_read_readvariableop>
:savev2_nadam_rot_equiv_conv2d_1_bias_m_read_readvariableop@
<savev2_nadam_rot_equiv_conv2d_2_kernel_m_read_readvariableop>
:savev2_nadam_rot_equiv_conv2d_2_bias_m_read_readvariableop@
<savev2_nadam_rot_equiv_conv2d_3_kernel_m_read_readvariableop>
:savev2_nadam_rot_equiv_conv2d_3_bias_m_read_readvariableop@
<savev2_nadam_rot_equiv_conv2d_4_kernel_m_read_readvariableop>
:savev2_nadam_rot_equiv_conv2d_4_bias_m_read_readvariableop3
/savev2_nadam_dense_kernel_m_read_readvariableop1
-savev2_nadam_dense_bias_m_read_readvariableop5
1savev2_nadam_dense_1_kernel_m_read_readvariableop3
/savev2_nadam_dense_1_bias_m_read_readvariableop>
:savev2_nadam_rot_equiv_conv2d_kernel_v_read_readvariableop<
8savev2_nadam_rot_equiv_conv2d_bias_v_read_readvariableop@
<savev2_nadam_rot_equiv_conv2d_1_kernel_v_read_readvariableop>
:savev2_nadam_rot_equiv_conv2d_1_bias_v_read_readvariableop@
<savev2_nadam_rot_equiv_conv2d_2_kernel_v_read_readvariableop>
:savev2_nadam_rot_equiv_conv2d_2_bias_v_read_readvariableop@
<savev2_nadam_rot_equiv_conv2d_3_kernel_v_read_readvariableop>
:savev2_nadam_rot_equiv_conv2d_3_bias_v_read_readvariableop@
<savev2_nadam_rot_equiv_conv2d_4_kernel_v_read_readvariableop>
:savev2_nadam_rot_equiv_conv2d_4_bias_v_read_readvariableop3
/savev2_nadam_dense_kernel_v_read_readvariableop1
-savev2_nadam_dense_bias_v_read_readvariableop5
1savev2_nadam_dense_1_kernel_v_read_readvariableop3
/savev2_nadam_dense_1_bias_v_read_readvariableop
savev2_const

identity_1ИҐMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Т
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:5*
dtype0*ї
value±BЃ5B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH„
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:5*
dtype0*}
valuetBr5B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B и
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:02savev2_rot_equiv_conv2d_kernel_read_readvariableop0savev2_rot_equiv_conv2d_bias_read_readvariableop4savev2_rot_equiv_conv2d_1_kernel_read_readvariableop2savev2_rot_equiv_conv2d_1_bias_read_readvariableop4savev2_rot_equiv_conv2d_2_kernel_read_readvariableop2savev2_rot_equiv_conv2d_2_bias_read_readvariableop4savev2_rot_equiv_conv2d_3_kernel_read_readvariableop2savev2_rot_equiv_conv2d_3_bias_read_readvariableop4savev2_rot_equiv_conv2d_4_kernel_read_readvariableop2savev2_rot_equiv_conv2d_4_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop%savev2_nadam_iter_read_readvariableop'savev2_nadam_beta_1_read_readvariableop'savev2_nadam_beta_2_read_readvariableop&savev2_nadam_decay_read_readvariableop.savev2_nadam_learning_rate_read_readvariableop/savev2_nadam_momentum_cache_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop:savev2_nadam_rot_equiv_conv2d_kernel_m_read_readvariableop8savev2_nadam_rot_equiv_conv2d_bias_m_read_readvariableop<savev2_nadam_rot_equiv_conv2d_1_kernel_m_read_readvariableop:savev2_nadam_rot_equiv_conv2d_1_bias_m_read_readvariableop<savev2_nadam_rot_equiv_conv2d_2_kernel_m_read_readvariableop:savev2_nadam_rot_equiv_conv2d_2_bias_m_read_readvariableop<savev2_nadam_rot_equiv_conv2d_3_kernel_m_read_readvariableop:savev2_nadam_rot_equiv_conv2d_3_bias_m_read_readvariableop<savev2_nadam_rot_equiv_conv2d_4_kernel_m_read_readvariableop:savev2_nadam_rot_equiv_conv2d_4_bias_m_read_readvariableop/savev2_nadam_dense_kernel_m_read_readvariableop-savev2_nadam_dense_bias_m_read_readvariableop1savev2_nadam_dense_1_kernel_m_read_readvariableop/savev2_nadam_dense_1_bias_m_read_readvariableop:savev2_nadam_rot_equiv_conv2d_kernel_v_read_readvariableop8savev2_nadam_rot_equiv_conv2d_bias_v_read_readvariableop<savev2_nadam_rot_equiv_conv2d_1_kernel_v_read_readvariableop:savev2_nadam_rot_equiv_conv2d_1_bias_v_read_readvariableop<savev2_nadam_rot_equiv_conv2d_2_kernel_v_read_readvariableop:savev2_nadam_rot_equiv_conv2d_2_bias_v_read_readvariableop<savev2_nadam_rot_equiv_conv2d_3_kernel_v_read_readvariableop:savev2_nadam_rot_equiv_conv2d_3_bias_v_read_readvariableop<savev2_nadam_rot_equiv_conv2d_4_kernel_v_read_readvariableop:savev2_nadam_rot_equiv_conv2d_4_bias_v_read_readvariableop/savev2_nadam_dense_kernel_v_read_readvariableop-savev2_nadam_dense_bias_v_read_readvariableop1savev2_nadam_dense_1_kernel_v_read_readvariableop/savev2_nadam_dense_1_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *C
dtypes9
725	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Л
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*ю
_input_shapesм
й: : : :  : : @:@:@@:@:@А:А:	А : : :: : : : : : : : : : : : :  : : @:@:@@:@:@А:А:	А : : :: : :  : : @:@:@@:@:@А:А:	А : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:-	)
'
_output_shapes
:@А:!


_output_shapes	
:А:%!

_output_shapes
:	А : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:,(
&
_output_shapes
:@@:  

_output_shapes
:@:-!)
'
_output_shapes
:@А:!"

_output_shapes	
:А:%#!

_output_shapes
:	А : $

_output_shapes
: :$% 

_output_shapes

: : &

_output_shapes
::,'(
&
_output_shapes
: : (

_output_shapes
: :,)(
&
_output_shapes
:  : *

_output_shapes
: :,+(
&
_output_shapes
: @: ,

_output_shapes
:@:,-(
&
_output_shapes
:@@: .

_output_shapes
:@:-/)
'
_output_shapes
:@А:!0

_output_shapes	
:А:%1!

_output_shapes
:	А : 2

_output_shapes
: :$3 

_output_shapes

: : 4

_output_shapes
::5

_output_shapes
: 
∆6
j
N__inference_rot_equiv_pool2d_1_layer_call_and_return_conditional_losses_627499

inputs
identityG
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R K
ShapeShapeinputs*
T0*
_output_shapes
:*
out_type0	f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskG
sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RS
subSubstrided_slice:output:0sub/y:output:0*
T0	*
_output_shapes
: Z
clip_by_value/MinimumMinimumConst:output:0sub:z:0*
T0	*
_output_shapes
: Q
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R n
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*
_output_shapes
: X
GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€§
GatherV2GatherV2inputsclip_by_value:z:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€EE Ґ
max_pooling2d_1/MaxPoolMaxPoolGatherV2:output:0*/
_output_shapes
:€€€€€€€€€"" *
ksize
*
paddingVALID*
strides
I
Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_1Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_1Substrided_slice_1:output:0sub_1/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_1/MinimumMinimumConst_1:output:0	sub_1:z:0*
T0	*
_output_shapes
: S
clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€™

GatherV2_1GatherV2inputsclip_by_value_1:z:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€EE ¶
max_pooling2d_1/MaxPool_1MaxPoolGatherV2_1:output:0*/
_output_shapes
:€€€€€€€€€"" *
ksize
*
paddingVALID*
strides
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_2Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_2StridedSliceShape_2:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_2Substrided_slice_2:output:0sub_2/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_2/MinimumMinimumConst_2:output:0	sub_2:z:0*
T0	*
_output_shapes
: S
clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€™

GatherV2_2GatherV2inputsclip_by_value_2:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€EE ¶
max_pooling2d_1/MaxPool_2MaxPoolGatherV2_2:output:0*/
_output_shapes
:€€€€€€€€€"" *
ksize
*
paddingVALID*
strides
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_3Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€j
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_3StridedSliceShape_3:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_3Substrided_slice_3:output:0sub_3/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_3/MinimumMinimumConst_3:output:0	sub_3:z:0*
T0	*
_output_shapes
: S
clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_3Maximumclip_by_value_3/Minimum:z:0clip_by_value_3/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€™

GatherV2_3GatherV2inputsclip_by_value_3:z:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€EE ¶
max_pooling2d_1/MaxPool_3MaxPoolGatherV2_3:output:0*/
_output_shapes
:€€€€€€€€€"" *
ksize
*
paddingVALID*
strides
ч
stackPack max_pooling2d_1/MaxPool:output:0"max_pooling2d_1/MaxPool_1:output:0"max_pooling2d_1/MaxPool_2:output:0"max_pooling2d_1/MaxPool_3:output:0*
N*
T0*3
_output_shapes!
:€€€€€€€€€"" *
axisю€€€€€€€€b
IdentityIdentitystack:output:0*
T0*3
_output_shapes!
:€€€€€€€€€"" "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€EE :[ W
3
_output_shapes!
:€€€€€€€€€EE 
 
_user_specified_nameinputs
≥и
Ѓ
F__inference_sequential_layer_call_and_return_conditional_losses_627200

inputsN
4rot_equiv_conv2d_convolution_readvariableop_resource: >
0rot_equiv_conv2d_biasadd_readvariableop_resource: P
6rot_equiv_conv2d_1_convolution_readvariableop_resource:  @
2rot_equiv_conv2d_1_biasadd_readvariableop_resource: P
6rot_equiv_conv2d_2_convolution_readvariableop_resource: @@
2rot_equiv_conv2d_2_biasadd_readvariableop_resource:@P
6rot_equiv_conv2d_3_convolution_readvariableop_resource:@@@
2rot_equiv_conv2d_3_biasadd_readvariableop_resource:@Q
6rot_equiv_conv2d_4_convolution_readvariableop_resource:@АA
2rot_equiv_conv2d_4_biasadd_readvariableop_resource:	А7
$dense_matmul_readvariableop_resource:	А 3
%dense_biasadd_readvariableop_resource: 8
&dense_1_matmul_readvariableop_resource: 5
'dense_1_biasadd_readvariableop_resource:
identityИҐdense/BiasAdd/ReadVariableOpҐdense/MatMul/ReadVariableOpҐdense_1/BiasAdd/ReadVariableOpҐdense_1/MatMul/ReadVariableOpҐ'rot_equiv_conv2d/BiasAdd/ReadVariableOpҐrot_equiv_conv2d/ReadVariableOpҐ!rot_equiv_conv2d/ReadVariableOp_1Ґ!rot_equiv_conv2d/ReadVariableOp_2Ґ+rot_equiv_conv2d/convolution/ReadVariableOpҐ)rot_equiv_conv2d_1/BiasAdd/ReadVariableOpҐ-rot_equiv_conv2d_1/convolution/ReadVariableOpҐ/rot_equiv_conv2d_1/convolution_1/ReadVariableOpҐ/rot_equiv_conv2d_1/convolution_2/ReadVariableOpҐ/rot_equiv_conv2d_1/convolution_3/ReadVariableOpҐ)rot_equiv_conv2d_2/BiasAdd/ReadVariableOpҐ-rot_equiv_conv2d_2/convolution/ReadVariableOpҐ/rot_equiv_conv2d_2/convolution_1/ReadVariableOpҐ/rot_equiv_conv2d_2/convolution_2/ReadVariableOpҐ/rot_equiv_conv2d_2/convolution_3/ReadVariableOpҐ)rot_equiv_conv2d_3/BiasAdd/ReadVariableOpҐ-rot_equiv_conv2d_3/convolution/ReadVariableOpҐ/rot_equiv_conv2d_3/convolution_1/ReadVariableOpҐ/rot_equiv_conv2d_3/convolution_2/ReadVariableOpҐ/rot_equiv_conv2d_3/convolution_3/ReadVariableOpҐ)rot_equiv_conv2d_4/BiasAdd/ReadVariableOpҐ-rot_equiv_conv2d_4/convolution/ReadVariableOpҐ/rot_equiv_conv2d_4/convolution_1/ReadVariableOpҐ/rot_equiv_conv2d_4/convolution_2/ReadVariableOpҐ/rot_equiv_conv2d_4/convolution_3/ReadVariableOp®
+rot_equiv_conv2d/convolution/ReadVariableOpReadVariableOp4rot_equiv_conv2d_convolution_readvariableop_resource*&
_output_shapes
: *
dtype0»
rot_equiv_conv2d/convolutionConv2Dinputs3rot_equiv_conv2d/convolution/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ОО *
paddingVALID*
strides
°
$rot_equiv_conv2d/Rank/ReadVariableOpReadVariableOp4rot_equiv_conv2d_convolution_readvariableop_resource*&
_output_shapes
: *
dtype0W
rot_equiv_conv2d/RankConst*
_output_shapes
: *
dtype0*
value	B :^
rot_equiv_conv2d/range/startConst*
_output_shapes
: *
dtype0*
value	B : ^
rot_equiv_conv2d/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :©
rot_equiv_conv2d/rangeRange%rot_equiv_conv2d/range/start:output:0rot_equiv_conv2d/Rank:output:0%rot_equiv_conv2d/range/delta:output:0*
_output_shapes
:Е
,rot_equiv_conv2d/TensorScatterUpdate/indicesConst*
_output_shapes

:*
dtype0*!
valueB"       }
,rot_equiv_conv2d/TensorScatterUpdate/updatesConst*
_output_shapes
:*
dtype0*
valueB"       €
$rot_equiv_conv2d/TensorScatterUpdateTensorScatterUpdaterot_equiv_conv2d/range:output:05rot_equiv_conv2d/TensorScatterUpdate/indices:output:05rot_equiv_conv2d/TensorScatterUpdate/updates:output:0*
T0*
Tindices0*
_output_shapes
:Ь
rot_equiv_conv2d/ReadVariableOpReadVariableOp4rot_equiv_conv2d_convolution_readvariableop_resource*&
_output_shapes
: *
dtype0Y
rot_equiv_conv2d/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :i
rot_equiv_conv2d/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:Ђ
rot_equiv_conv2d/ReverseV2	ReverseV2'rot_equiv_conv2d/ReadVariableOp:value:0(rot_equiv_conv2d/ReverseV2/axis:output:0*
T0*&
_output_shapes
: ђ
rot_equiv_conv2d/transpose	Transpose#rot_equiv_conv2d/ReverseV2:output:0-rot_equiv_conv2d/TensorScatterUpdate:output:0*
T0*&
_output_shapes
: µ
rot_equiv_conv2d/convolution_1Conv2Dinputsrot_equiv_conv2d/transpose:y:0*
T0*1
_output_shapes
:€€€€€€€€€ОО *
paddingVALID*
strides
Y
rot_equiv_conv2d/Rank_2Const*
_output_shapes
: *
dtype0*
value	B :`
rot_equiv_conv2d/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : `
rot_equiv_conv2d/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :±
rot_equiv_conv2d/range_1Range'rot_equiv_conv2d/range_1/start:output:0 rot_equiv_conv2d/Rank_2:output:0'rot_equiv_conv2d/range_1/delta:output:0*
_output_shapes
:З
.rot_equiv_conv2d/TensorScatterUpdate_1/indicesConst*
_output_shapes

:*
dtype0*!
valueB"      
.rot_equiv_conv2d/TensorScatterUpdate_1/updatesConst*
_output_shapes
:*
dtype0*
valueB"      З
&rot_equiv_conv2d/TensorScatterUpdate_1TensorScatterUpdate!rot_equiv_conv2d/range_1:output:07rot_equiv_conv2d/TensorScatterUpdate_1/indices:output:07rot_equiv_conv2d/TensorScatterUpdate_1/updates:output:0*
T0*
Tindices0*
_output_shapes
:њ
rot_equiv_conv2d/transpose_1	Transpose'rot_equiv_conv2d/convolution_1:output:0/rot_equiv_conv2d/TensorScatterUpdate_1:output:0*
T0*1
_output_shapes
:€€€€€€€€€ОО Y
rot_equiv_conv2d/Rank_3Const*
_output_shapes
: *
dtype0*
value	B :k
!rot_equiv_conv2d/ReverseV2_1/axisConst*
_output_shapes
:*
dtype0*
valueB:≥
rot_equiv_conv2d/ReverseV2_1	ReverseV2 rot_equiv_conv2d/transpose_1:y:0*rot_equiv_conv2d/ReverseV2_1/axis:output:0*
T0*1
_output_shapes
:€€€€€€€€€ОО £
&rot_equiv_conv2d/Rank_4/ReadVariableOpReadVariableOp4rot_equiv_conv2d_convolution_readvariableop_resource*&
_output_shapes
: *
dtype0Y
rot_equiv_conv2d/Rank_4Const*
_output_shapes
: *
dtype0*
value	B :Ю
!rot_equiv_conv2d/ReadVariableOp_1ReadVariableOp4rot_equiv_conv2d_convolution_readvariableop_resource*&
_output_shapes
: *
dtype0Y
rot_equiv_conv2d/Rank_5Const*
_output_shapes
: *
dtype0*
value	B :k
!rot_equiv_conv2d/ReverseV2_2/axisConst*
_output_shapes
:*
dtype0*
valueB: ±
rot_equiv_conv2d/ReverseV2_2	ReverseV2)rot_equiv_conv2d/ReadVariableOp_1:value:0*rot_equiv_conv2d/ReverseV2_2/axis:output:0*
T0*&
_output_shapes
: Y
rot_equiv_conv2d/Rank_6Const*
_output_shapes
: *
dtype0*
value	B :k
!rot_equiv_conv2d/ReverseV2_3/axisConst*
_output_shapes
:*
dtype0*
valueB:≠
rot_equiv_conv2d/ReverseV2_3	ReverseV2%rot_equiv_conv2d/ReverseV2_2:output:0*rot_equiv_conv2d/ReverseV2_3/axis:output:0*
T0*&
_output_shapes
: Љ
rot_equiv_conv2d/convolution_2Conv2Dinputs%rot_equiv_conv2d/ReverseV2_3:output:0*
T0*1
_output_shapes
:€€€€€€€€€ОО *
paddingVALID*
strides
Y
rot_equiv_conv2d/Rank_7Const*
_output_shapes
: *
dtype0*
value	B :Y
rot_equiv_conv2d/Rank_8Const*
_output_shapes
: *
dtype0*
value	B :k
!rot_equiv_conv2d/ReverseV2_4/axisConst*
_output_shapes
:*
dtype0*
valueB:Ї
rot_equiv_conv2d/ReverseV2_4	ReverseV2'rot_equiv_conv2d/convolution_2:output:0*rot_equiv_conv2d/ReverseV2_4/axis:output:0*
T0*1
_output_shapes
:€€€€€€€€€ОО Y
rot_equiv_conv2d/Rank_9Const*
_output_shapes
: *
dtype0*
value	B :k
!rot_equiv_conv2d/ReverseV2_5/axisConst*
_output_shapes
:*
dtype0*
valueB:Є
rot_equiv_conv2d/ReverseV2_5	ReverseV2%rot_equiv_conv2d/ReverseV2_4:output:0*rot_equiv_conv2d/ReverseV2_5/axis:output:0*
T0*1
_output_shapes
:€€€€€€€€€ОО §
'rot_equiv_conv2d/Rank_10/ReadVariableOpReadVariableOp4rot_equiv_conv2d_convolution_readvariableop_resource*&
_output_shapes
: *
dtype0Z
rot_equiv_conv2d/Rank_10Const*
_output_shapes
: *
dtype0*
value	B :`
rot_equiv_conv2d/range_2/startConst*
_output_shapes
: *
dtype0*
value	B : `
rot_equiv_conv2d/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :≤
rot_equiv_conv2d/range_2Range'rot_equiv_conv2d/range_2/start:output:0!rot_equiv_conv2d/Rank_10:output:0'rot_equiv_conv2d/range_2/delta:output:0*
_output_shapes
:З
.rot_equiv_conv2d/TensorScatterUpdate_2/indicesConst*
_output_shapes

:*
dtype0*!
valueB"       
.rot_equiv_conv2d/TensorScatterUpdate_2/updatesConst*
_output_shapes
:*
dtype0*
valueB"       З
&rot_equiv_conv2d/TensorScatterUpdate_2TensorScatterUpdate!rot_equiv_conv2d/range_2:output:07rot_equiv_conv2d/TensorScatterUpdate_2/indices:output:07rot_equiv_conv2d/TensorScatterUpdate_2/updates:output:0*
T0*
Tindices0*
_output_shapes
:Ю
!rot_equiv_conv2d/ReadVariableOp_2ReadVariableOp4rot_equiv_conv2d_convolution_readvariableop_resource*&
_output_shapes
: *
dtype0ґ
rot_equiv_conv2d/transpose_2	Transpose)rot_equiv_conv2d/ReadVariableOp_2:value:0/rot_equiv_conv2d/TensorScatterUpdate_2:output:0*
T0*&
_output_shapes
: Z
rot_equiv_conv2d/Rank_11Const*
_output_shapes
: *
dtype0*
value	B :k
!rot_equiv_conv2d/ReverseV2_6/axisConst*
_output_shapes
:*
dtype0*
valueB:®
rot_equiv_conv2d/ReverseV2_6	ReverseV2 rot_equiv_conv2d/transpose_2:y:0*rot_equiv_conv2d/ReverseV2_6/axis:output:0*
T0*&
_output_shapes
: Љ
rot_equiv_conv2d/convolution_3Conv2Dinputs%rot_equiv_conv2d/ReverseV2_6:output:0*
T0*1
_output_shapes
:€€€€€€€€€ОО *
paddingVALID*
strides
Z
rot_equiv_conv2d/Rank_12Const*
_output_shapes
: *
dtype0*
value	B :`
rot_equiv_conv2d/range_3/startConst*
_output_shapes
: *
dtype0*
value	B : `
rot_equiv_conv2d/range_3/deltaConst*
_output_shapes
: *
dtype0*
value	B :≤
rot_equiv_conv2d/range_3Range'rot_equiv_conv2d/range_3/start:output:0!rot_equiv_conv2d/Rank_12:output:0'rot_equiv_conv2d/range_3/delta:output:0*
_output_shapes
:З
.rot_equiv_conv2d/TensorScatterUpdate_3/indicesConst*
_output_shapes

:*
dtype0*!
valueB"      
.rot_equiv_conv2d/TensorScatterUpdate_3/updatesConst*
_output_shapes
:*
dtype0*
valueB"      З
&rot_equiv_conv2d/TensorScatterUpdate_3TensorScatterUpdate!rot_equiv_conv2d/range_3:output:07rot_equiv_conv2d/TensorScatterUpdate_3/indices:output:07rot_equiv_conv2d/TensorScatterUpdate_3/updates:output:0*
T0*
Tindices0*
_output_shapes
:Z
rot_equiv_conv2d/Rank_13Const*
_output_shapes
: *
dtype0*
value	B :k
!rot_equiv_conv2d/ReverseV2_7/axisConst*
_output_shapes
:*
dtype0*
valueB:Ї
rot_equiv_conv2d/ReverseV2_7	ReverseV2'rot_equiv_conv2d/convolution_3:output:0*rot_equiv_conv2d/ReverseV2_7/axis:output:0*
T0*1
_output_shapes
:€€€€€€€€€ОО љ
rot_equiv_conv2d/transpose_3	Transpose%rot_equiv_conv2d/ReverseV2_7:output:0/rot_equiv_conv2d/TensorScatterUpdate_3:output:0*
T0*1
_output_shapes
:€€€€€€€€€ОО У
rot_equiv_conv2d/stackPack%rot_equiv_conv2d/convolution:output:0%rot_equiv_conv2d/ReverseV2_1:output:0%rot_equiv_conv2d/ReverseV2_5:output:0 rot_equiv_conv2d/transpose_3:y:0*
N*
T0*5
_output_shapes#
!:€€€€€€€€€ОО *
axisю€€€€€€€€~
rot_equiv_conv2d/ReluRelurot_equiv_conv2d/stack:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€ОО Ф
'rot_equiv_conv2d/BiasAdd/ReadVariableOpReadVariableOp0rot_equiv_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0є
rot_equiv_conv2d/BiasAddBiasAdd#rot_equiv_conv2d/Relu:activations:0/rot_equiv_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:€€€€€€€€€ОО X
rot_equiv_pool2d/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R w
rot_equiv_pool2d/ShapeShape!rot_equiv_conv2d/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	w
$rot_equiv_pool2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€y
&rot_equiv_pool2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€p
&rot_equiv_pool2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¶
rot_equiv_pool2d/strided_sliceStridedSlicerot_equiv_pool2d/Shape:output:0-rot_equiv_pool2d/strided_slice/stack:output:0/rot_equiv_pool2d/strided_slice/stack_1:output:0/rot_equiv_pool2d/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskX
rot_equiv_pool2d/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RЖ
rot_equiv_pool2d/subSub'rot_equiv_pool2d/strided_slice:output:0rot_equiv_pool2d/sub/y:output:0*
T0	*
_output_shapes
: Н
&rot_equiv_pool2d/clip_by_value/MinimumMinimumrot_equiv_pool2d/Const:output:0rot_equiv_pool2d/sub:z:0*
T0	*
_output_shapes
: b
 rot_equiv_pool2d/clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R °
rot_equiv_pool2d/clip_by_valueMaximum*rot_equiv_pool2d/clip_by_value/Minimum:z:0)rot_equiv_pool2d/clip_by_value/y:output:0*
T0	*
_output_shapes
: i
rot_equiv_pool2d/GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ф
rot_equiv_pool2d/GatherV2GatherV2!rot_equiv_conv2d/BiasAdd:output:0"rot_equiv_pool2d/clip_by_value:z:0'rot_equiv_pool2d/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*1
_output_shapes
:€€€€€€€€€ОО ¬
&rot_equiv_pool2d/max_pooling2d/MaxPoolMaxPool"rot_equiv_pool2d/GatherV2:output:0*/
_output_shapes
:€€€€€€€€€GG *
ksize
*
paddingVALID*
strides
Z
rot_equiv_pool2d/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 Ry
rot_equiv_pool2d/Shape_1Shape!rot_equiv_conv2d/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	y
&rot_equiv_pool2d/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€{
(rot_equiv_pool2d/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€r
(rot_equiv_pool2d/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:∞
 rot_equiv_pool2d/strided_slice_1StridedSlice!rot_equiv_pool2d/Shape_1:output:0/rot_equiv_pool2d/strided_slice_1/stack:output:01rot_equiv_pool2d/strided_slice_1/stack_1:output:01rot_equiv_pool2d/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskZ
rot_equiv_pool2d/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 RМ
rot_equiv_pool2d/sub_1Sub)rot_equiv_pool2d/strided_slice_1:output:0!rot_equiv_pool2d/sub_1/y:output:0*
T0	*
_output_shapes
: У
(rot_equiv_pool2d/clip_by_value_1/MinimumMinimum!rot_equiv_pool2d/Const_1:output:0rot_equiv_pool2d/sub_1:z:0*
T0	*
_output_shapes
: d
"rot_equiv_pool2d/clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R І
 rot_equiv_pool2d/clip_by_value_1Maximum,rot_equiv_pool2d/clip_by_value_1/Minimum:z:0+rot_equiv_pool2d/clip_by_value_1/y:output:0*
T0	*
_output_shapes
: k
 rot_equiv_pool2d/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ъ
rot_equiv_pool2d/GatherV2_1GatherV2!rot_equiv_conv2d/BiasAdd:output:0$rot_equiv_pool2d/clip_by_value_1:z:0)rot_equiv_pool2d/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*1
_output_shapes
:€€€€€€€€€ОО ∆
(rot_equiv_pool2d/max_pooling2d/MaxPool_1MaxPool$rot_equiv_pool2d/GatherV2_1:output:0*/
_output_shapes
:€€€€€€€€€GG *
ksize
*
paddingVALID*
strides
Z
rot_equiv_pool2d/Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 Ry
rot_equiv_pool2d/Shape_2Shape!rot_equiv_conv2d/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	y
&rot_equiv_pool2d/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€{
(rot_equiv_pool2d/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€r
(rot_equiv_pool2d/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:∞
 rot_equiv_pool2d/strided_slice_2StridedSlice!rot_equiv_pool2d/Shape_2:output:0/rot_equiv_pool2d/strided_slice_2/stack:output:01rot_equiv_pool2d/strided_slice_2/stack_1:output:01rot_equiv_pool2d/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskZ
rot_equiv_pool2d/sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 RМ
rot_equiv_pool2d/sub_2Sub)rot_equiv_pool2d/strided_slice_2:output:0!rot_equiv_pool2d/sub_2/y:output:0*
T0	*
_output_shapes
: У
(rot_equiv_pool2d/clip_by_value_2/MinimumMinimum!rot_equiv_pool2d/Const_2:output:0rot_equiv_pool2d/sub_2:z:0*
T0	*
_output_shapes
: d
"rot_equiv_pool2d/clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R І
 rot_equiv_pool2d/clip_by_value_2Maximum,rot_equiv_pool2d/clip_by_value_2/Minimum:z:0+rot_equiv_pool2d/clip_by_value_2/y:output:0*
T0	*
_output_shapes
: k
 rot_equiv_pool2d/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ъ
rot_equiv_pool2d/GatherV2_2GatherV2!rot_equiv_conv2d/BiasAdd:output:0$rot_equiv_pool2d/clip_by_value_2:z:0)rot_equiv_pool2d/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*1
_output_shapes
:€€€€€€€€€ОО ∆
(rot_equiv_pool2d/max_pooling2d/MaxPool_2MaxPool$rot_equiv_pool2d/GatherV2_2:output:0*/
_output_shapes
:€€€€€€€€€GG *
ksize
*
paddingVALID*
strides
Z
rot_equiv_pool2d/Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 Ry
rot_equiv_pool2d/Shape_3Shape!rot_equiv_conv2d/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	y
&rot_equiv_pool2d/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€{
(rot_equiv_pool2d/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€r
(rot_equiv_pool2d/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:∞
 rot_equiv_pool2d/strided_slice_3StridedSlice!rot_equiv_pool2d/Shape_3:output:0/rot_equiv_pool2d/strided_slice_3/stack:output:01rot_equiv_pool2d/strided_slice_3/stack_1:output:01rot_equiv_pool2d/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskZ
rot_equiv_pool2d/sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 RМ
rot_equiv_pool2d/sub_3Sub)rot_equiv_pool2d/strided_slice_3:output:0!rot_equiv_pool2d/sub_3/y:output:0*
T0	*
_output_shapes
: У
(rot_equiv_pool2d/clip_by_value_3/MinimumMinimum!rot_equiv_pool2d/Const_3:output:0rot_equiv_pool2d/sub_3:z:0*
T0	*
_output_shapes
: d
"rot_equiv_pool2d/clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R І
 rot_equiv_pool2d/clip_by_value_3Maximum,rot_equiv_pool2d/clip_by_value_3/Minimum:z:0+rot_equiv_pool2d/clip_by_value_3/y:output:0*
T0	*
_output_shapes
: k
 rot_equiv_pool2d/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ъ
rot_equiv_pool2d/GatherV2_3GatherV2!rot_equiv_conv2d/BiasAdd:output:0$rot_equiv_pool2d/clip_by_value_3:z:0)rot_equiv_pool2d/GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*1
_output_shapes
:€€€€€€€€€ОО ∆
(rot_equiv_pool2d/max_pooling2d/MaxPool_3MaxPool$rot_equiv_pool2d/GatherV2_3:output:0*/
_output_shapes
:€€€€€€€€€GG *
ksize
*
paddingVALID*
strides
ƒ
rot_equiv_pool2d/stackPack/rot_equiv_pool2d/max_pooling2d/MaxPool:output:01rot_equiv_pool2d/max_pooling2d/MaxPool_1:output:01rot_equiv_pool2d/max_pooling2d/MaxPool_2:output:01rot_equiv_pool2d/max_pooling2d/MaxPool_3:output:0*
N*
T0*3
_output_shapes!
:€€€€€€€€€GG *
axisю€€€€€€€€Z
rot_equiv_conv2d_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R w
rot_equiv_conv2d_1/ShapeShaperot_equiv_pool2d/stack:output:0*
T0*
_output_shapes
:*
out_type0	y
&rot_equiv_conv2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€{
(rot_equiv_conv2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€r
(rot_equiv_conv2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:∞
 rot_equiv_conv2d_1/strided_sliceStridedSlice!rot_equiv_conv2d_1/Shape:output:0/rot_equiv_conv2d_1/strided_slice/stack:output:01rot_equiv_conv2d_1/strided_slice/stack_1:output:01rot_equiv_conv2d_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskZ
rot_equiv_conv2d_1/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RМ
rot_equiv_conv2d_1/subSub)rot_equiv_conv2d_1/strided_slice:output:0!rot_equiv_conv2d_1/sub/y:output:0*
T0	*
_output_shapes
: У
(rot_equiv_conv2d_1/clip_by_value/MinimumMinimum!rot_equiv_conv2d_1/Const:output:0rot_equiv_conv2d_1/sub:z:0*
T0	*
_output_shapes
: d
"rot_equiv_conv2d_1/clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R І
 rot_equiv_conv2d_1/clip_by_valueMaximum,rot_equiv_conv2d_1/clip_by_value/Minimum:z:0+rot_equiv_conv2d_1/clip_by_value/y:output:0*
T0	*
_output_shapes
: k
 rot_equiv_conv2d_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ц
rot_equiv_conv2d_1/GatherV2GatherV2rot_equiv_pool2d/stack:output:0$rot_equiv_conv2d_1/clip_by_value:z:0)rot_equiv_conv2d_1/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€GG ђ
-rot_equiv_conv2d_1/convolution/ReadVariableOpReadVariableOp6rot_equiv_conv2d_1_convolution_readvariableop_resource*&
_output_shapes
:  *
dtype0и
rot_equiv_conv2d_1/convolutionConv2D$rot_equiv_conv2d_1/GatherV2:output:05rot_equiv_conv2d_1/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€EE *
paddingVALID*
strides
\
rot_equiv_conv2d_1/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 Ry
rot_equiv_conv2d_1/Shape_1Shaperot_equiv_pool2d/stack:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_conv2d_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€}
*rot_equiv_conv2d_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€t
*rot_equiv_conv2d_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ї
"rot_equiv_conv2d_1/strided_slice_1StridedSlice#rot_equiv_conv2d_1/Shape_1:output:01rot_equiv_conv2d_1/strided_slice_1/stack:output:03rot_equiv_conv2d_1/strided_slice_1/stack_1:output:03rot_equiv_conv2d_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_conv2d_1/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 RТ
rot_equiv_conv2d_1/sub_1Sub+rot_equiv_conv2d_1/strided_slice_1:output:0#rot_equiv_conv2d_1/sub_1/y:output:0*
T0	*
_output_shapes
: Щ
*rot_equiv_conv2d_1/clip_by_value_1/MinimumMinimum#rot_equiv_conv2d_1/Const_1:output:0rot_equiv_conv2d_1/sub_1:z:0*
T0	*
_output_shapes
: f
$rot_equiv_conv2d_1/clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ≠
"rot_equiv_conv2d_1/clip_by_value_1Maximum.rot_equiv_conv2d_1/clip_by_value_1/Minimum:z:0-rot_equiv_conv2d_1/clip_by_value_1/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_conv2d_1/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ь
rot_equiv_conv2d_1/GatherV2_1GatherV2rot_equiv_pool2d/stack:output:0&rot_equiv_conv2d_1/clip_by_value_1:z:0+rot_equiv_conv2d_1/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€GG Ѓ
/rot_equiv_conv2d_1/convolution_1/ReadVariableOpReadVariableOp6rot_equiv_conv2d_1_convolution_readvariableop_resource*&
_output_shapes
:  *
dtype0о
 rot_equiv_conv2d_1/convolution_1Conv2D&rot_equiv_conv2d_1/GatherV2_1:output:07rot_equiv_conv2d_1/convolution_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€EE *
paddingVALID*
strides
\
rot_equiv_conv2d_1/Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 Ry
rot_equiv_conv2d_1/Shape_2Shaperot_equiv_pool2d/stack:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_conv2d_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€}
*rot_equiv_conv2d_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€t
*rot_equiv_conv2d_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ї
"rot_equiv_conv2d_1/strided_slice_2StridedSlice#rot_equiv_conv2d_1/Shape_2:output:01rot_equiv_conv2d_1/strided_slice_2/stack:output:03rot_equiv_conv2d_1/strided_slice_2/stack_1:output:03rot_equiv_conv2d_1/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_conv2d_1/sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 RТ
rot_equiv_conv2d_1/sub_2Sub+rot_equiv_conv2d_1/strided_slice_2:output:0#rot_equiv_conv2d_1/sub_2/y:output:0*
T0	*
_output_shapes
: Щ
*rot_equiv_conv2d_1/clip_by_value_2/MinimumMinimum#rot_equiv_conv2d_1/Const_2:output:0rot_equiv_conv2d_1/sub_2:z:0*
T0	*
_output_shapes
: f
$rot_equiv_conv2d_1/clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ≠
"rot_equiv_conv2d_1/clip_by_value_2Maximum.rot_equiv_conv2d_1/clip_by_value_2/Minimum:z:0-rot_equiv_conv2d_1/clip_by_value_2/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_conv2d_1/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ь
rot_equiv_conv2d_1/GatherV2_2GatherV2rot_equiv_pool2d/stack:output:0&rot_equiv_conv2d_1/clip_by_value_2:z:0+rot_equiv_conv2d_1/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€GG Ѓ
/rot_equiv_conv2d_1/convolution_2/ReadVariableOpReadVariableOp6rot_equiv_conv2d_1_convolution_readvariableop_resource*&
_output_shapes
:  *
dtype0о
 rot_equiv_conv2d_1/convolution_2Conv2D&rot_equiv_conv2d_1/GatherV2_2:output:07rot_equiv_conv2d_1/convolution_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€EE *
paddingVALID*
strides
\
rot_equiv_conv2d_1/Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 Ry
rot_equiv_conv2d_1/Shape_3Shaperot_equiv_pool2d/stack:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_conv2d_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€}
*rot_equiv_conv2d_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€t
*rot_equiv_conv2d_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ї
"rot_equiv_conv2d_1/strided_slice_3StridedSlice#rot_equiv_conv2d_1/Shape_3:output:01rot_equiv_conv2d_1/strided_slice_3/stack:output:03rot_equiv_conv2d_1/strided_slice_3/stack_1:output:03rot_equiv_conv2d_1/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_conv2d_1/sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 RТ
rot_equiv_conv2d_1/sub_3Sub+rot_equiv_conv2d_1/strided_slice_3:output:0#rot_equiv_conv2d_1/sub_3/y:output:0*
T0	*
_output_shapes
: Щ
*rot_equiv_conv2d_1/clip_by_value_3/MinimumMinimum#rot_equiv_conv2d_1/Const_3:output:0rot_equiv_conv2d_1/sub_3:z:0*
T0	*
_output_shapes
: f
$rot_equiv_conv2d_1/clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ≠
"rot_equiv_conv2d_1/clip_by_value_3Maximum.rot_equiv_conv2d_1/clip_by_value_3/Minimum:z:0-rot_equiv_conv2d_1/clip_by_value_3/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_conv2d_1/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ь
rot_equiv_conv2d_1/GatherV2_3GatherV2rot_equiv_pool2d/stack:output:0&rot_equiv_conv2d_1/clip_by_value_3:z:0+rot_equiv_conv2d_1/GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€GG Ѓ
/rot_equiv_conv2d_1/convolution_3/ReadVariableOpReadVariableOp6rot_equiv_conv2d_1_convolution_readvariableop_resource*&
_output_shapes
:  *
dtype0о
 rot_equiv_conv2d_1/convolution_3Conv2D&rot_equiv_conv2d_1/GatherV2_3:output:07rot_equiv_conv2d_1/convolution_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€EE *
paddingVALID*
strides
¶
rot_equiv_conv2d_1/stackPack'rot_equiv_conv2d_1/convolution:output:0)rot_equiv_conv2d_1/convolution_1:output:0)rot_equiv_conv2d_1/convolution_2:output:0)rot_equiv_conv2d_1/convolution_3:output:0*
N*
T0*3
_output_shapes!
:€€€€€€€€€EE *
axisю€€€€€€€€А
rot_equiv_conv2d_1/ReluRelu!rot_equiv_conv2d_1/stack:output:0*
T0*3
_output_shapes!
:€€€€€€€€€EE Ш
)rot_equiv_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp2rot_equiv_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0љ
rot_equiv_conv2d_1/BiasAddBiasAdd%rot_equiv_conv2d_1/Relu:activations:01rot_equiv_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€EE Z
rot_equiv_pool2d_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R {
rot_equiv_pool2d_1/ShapeShape#rot_equiv_conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	y
&rot_equiv_pool2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€{
(rot_equiv_pool2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€r
(rot_equiv_pool2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:∞
 rot_equiv_pool2d_1/strided_sliceStridedSlice!rot_equiv_pool2d_1/Shape:output:0/rot_equiv_pool2d_1/strided_slice/stack:output:01rot_equiv_pool2d_1/strided_slice/stack_1:output:01rot_equiv_pool2d_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskZ
rot_equiv_pool2d_1/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RМ
rot_equiv_pool2d_1/subSub)rot_equiv_pool2d_1/strided_slice:output:0!rot_equiv_pool2d_1/sub/y:output:0*
T0	*
_output_shapes
: У
(rot_equiv_pool2d_1/clip_by_value/MinimumMinimum!rot_equiv_pool2d_1/Const:output:0rot_equiv_pool2d_1/sub:z:0*
T0	*
_output_shapes
: d
"rot_equiv_pool2d_1/clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R І
 rot_equiv_pool2d_1/clip_by_valueMaximum,rot_equiv_pool2d_1/clip_by_value/Minimum:z:0+rot_equiv_pool2d_1/clip_by_value/y:output:0*
T0	*
_output_shapes
: k
 rot_equiv_pool2d_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ъ
rot_equiv_pool2d_1/GatherV2GatherV2#rot_equiv_conv2d_1/BiasAdd:output:0$rot_equiv_pool2d_1/clip_by_value:z:0)rot_equiv_pool2d_1/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€EE »
*rot_equiv_pool2d_1/max_pooling2d_1/MaxPoolMaxPool$rot_equiv_pool2d_1/GatherV2:output:0*/
_output_shapes
:€€€€€€€€€"" *
ksize
*
paddingVALID*
strides
\
rot_equiv_pool2d_1/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 R}
rot_equiv_pool2d_1/Shape_1Shape#rot_equiv_conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_pool2d_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€}
*rot_equiv_pool2d_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€t
*rot_equiv_pool2d_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ї
"rot_equiv_pool2d_1/strided_slice_1StridedSlice#rot_equiv_pool2d_1/Shape_1:output:01rot_equiv_pool2d_1/strided_slice_1/stack:output:03rot_equiv_pool2d_1/strided_slice_1/stack_1:output:03rot_equiv_pool2d_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_pool2d_1/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 RТ
rot_equiv_pool2d_1/sub_1Sub+rot_equiv_pool2d_1/strided_slice_1:output:0#rot_equiv_pool2d_1/sub_1/y:output:0*
T0	*
_output_shapes
: Щ
*rot_equiv_pool2d_1/clip_by_value_1/MinimumMinimum#rot_equiv_pool2d_1/Const_1:output:0rot_equiv_pool2d_1/sub_1:z:0*
T0	*
_output_shapes
: f
$rot_equiv_pool2d_1/clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ≠
"rot_equiv_pool2d_1/clip_by_value_1Maximum.rot_equiv_pool2d_1/clip_by_value_1/Minimum:z:0-rot_equiv_pool2d_1/clip_by_value_1/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_pool2d_1/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€А
rot_equiv_pool2d_1/GatherV2_1GatherV2#rot_equiv_conv2d_1/BiasAdd:output:0&rot_equiv_pool2d_1/clip_by_value_1:z:0+rot_equiv_pool2d_1/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€EE ћ
,rot_equiv_pool2d_1/max_pooling2d_1/MaxPool_1MaxPool&rot_equiv_pool2d_1/GatherV2_1:output:0*/
_output_shapes
:€€€€€€€€€"" *
ksize
*
paddingVALID*
strides
\
rot_equiv_pool2d_1/Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R}
rot_equiv_pool2d_1/Shape_2Shape#rot_equiv_conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_pool2d_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€}
*rot_equiv_pool2d_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€t
*rot_equiv_pool2d_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ї
"rot_equiv_pool2d_1/strided_slice_2StridedSlice#rot_equiv_pool2d_1/Shape_2:output:01rot_equiv_pool2d_1/strided_slice_2/stack:output:03rot_equiv_pool2d_1/strided_slice_2/stack_1:output:03rot_equiv_pool2d_1/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_pool2d_1/sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 RТ
rot_equiv_pool2d_1/sub_2Sub+rot_equiv_pool2d_1/strided_slice_2:output:0#rot_equiv_pool2d_1/sub_2/y:output:0*
T0	*
_output_shapes
: Щ
*rot_equiv_pool2d_1/clip_by_value_2/MinimumMinimum#rot_equiv_pool2d_1/Const_2:output:0rot_equiv_pool2d_1/sub_2:z:0*
T0	*
_output_shapes
: f
$rot_equiv_pool2d_1/clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ≠
"rot_equiv_pool2d_1/clip_by_value_2Maximum.rot_equiv_pool2d_1/clip_by_value_2/Minimum:z:0-rot_equiv_pool2d_1/clip_by_value_2/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_pool2d_1/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€А
rot_equiv_pool2d_1/GatherV2_2GatherV2#rot_equiv_conv2d_1/BiasAdd:output:0&rot_equiv_pool2d_1/clip_by_value_2:z:0+rot_equiv_pool2d_1/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€EE ћ
,rot_equiv_pool2d_1/max_pooling2d_1/MaxPool_2MaxPool&rot_equiv_pool2d_1/GatherV2_2:output:0*/
_output_shapes
:€€€€€€€€€"" *
ksize
*
paddingVALID*
strides
\
rot_equiv_pool2d_1/Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 R}
rot_equiv_pool2d_1/Shape_3Shape#rot_equiv_conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_pool2d_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€}
*rot_equiv_pool2d_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€t
*rot_equiv_pool2d_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ї
"rot_equiv_pool2d_1/strided_slice_3StridedSlice#rot_equiv_pool2d_1/Shape_3:output:01rot_equiv_pool2d_1/strided_slice_3/stack:output:03rot_equiv_pool2d_1/strided_slice_3/stack_1:output:03rot_equiv_pool2d_1/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_pool2d_1/sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 RТ
rot_equiv_pool2d_1/sub_3Sub+rot_equiv_pool2d_1/strided_slice_3:output:0#rot_equiv_pool2d_1/sub_3/y:output:0*
T0	*
_output_shapes
: Щ
*rot_equiv_pool2d_1/clip_by_value_3/MinimumMinimum#rot_equiv_pool2d_1/Const_3:output:0rot_equiv_pool2d_1/sub_3:z:0*
T0	*
_output_shapes
: f
$rot_equiv_pool2d_1/clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ≠
"rot_equiv_pool2d_1/clip_by_value_3Maximum.rot_equiv_pool2d_1/clip_by_value_3/Minimum:z:0-rot_equiv_pool2d_1/clip_by_value_3/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_pool2d_1/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€А
rot_equiv_pool2d_1/GatherV2_3GatherV2#rot_equiv_conv2d_1/BiasAdd:output:0&rot_equiv_pool2d_1/clip_by_value_3:z:0+rot_equiv_pool2d_1/GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€EE ћ
,rot_equiv_pool2d_1/max_pooling2d_1/MaxPool_3MaxPool&rot_equiv_pool2d_1/GatherV2_3:output:0*/
_output_shapes
:€€€€€€€€€"" *
ksize
*
paddingVALID*
strides
÷
rot_equiv_pool2d_1/stackPack3rot_equiv_pool2d_1/max_pooling2d_1/MaxPool:output:05rot_equiv_pool2d_1/max_pooling2d_1/MaxPool_1:output:05rot_equiv_pool2d_1/max_pooling2d_1/MaxPool_2:output:05rot_equiv_pool2d_1/max_pooling2d_1/MaxPool_3:output:0*
N*
T0*3
_output_shapes!
:€€€€€€€€€"" *
axisю€€€€€€€€Z
rot_equiv_conv2d_2/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R y
rot_equiv_conv2d_2/ShapeShape!rot_equiv_pool2d_1/stack:output:0*
T0*
_output_shapes
:*
out_type0	y
&rot_equiv_conv2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€{
(rot_equiv_conv2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€r
(rot_equiv_conv2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:∞
 rot_equiv_conv2d_2/strided_sliceStridedSlice!rot_equiv_conv2d_2/Shape:output:0/rot_equiv_conv2d_2/strided_slice/stack:output:01rot_equiv_conv2d_2/strided_slice/stack_1:output:01rot_equiv_conv2d_2/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskZ
rot_equiv_conv2d_2/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RМ
rot_equiv_conv2d_2/subSub)rot_equiv_conv2d_2/strided_slice:output:0!rot_equiv_conv2d_2/sub/y:output:0*
T0	*
_output_shapes
: У
(rot_equiv_conv2d_2/clip_by_value/MinimumMinimum!rot_equiv_conv2d_2/Const:output:0rot_equiv_conv2d_2/sub:z:0*
T0	*
_output_shapes
: d
"rot_equiv_conv2d_2/clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R І
 rot_equiv_conv2d_2/clip_by_valueMaximum,rot_equiv_conv2d_2/clip_by_value/Minimum:z:0+rot_equiv_conv2d_2/clip_by_value/y:output:0*
T0	*
_output_shapes
: k
 rot_equiv_conv2d_2/GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ш
rot_equiv_conv2d_2/GatherV2GatherV2!rot_equiv_pool2d_1/stack:output:0$rot_equiv_conv2d_2/clip_by_value:z:0)rot_equiv_conv2d_2/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€"" ђ
-rot_equiv_conv2d_2/convolution/ReadVariableOpReadVariableOp6rot_equiv_conv2d_2_convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0и
rot_equiv_conv2d_2/convolutionConv2D$rot_equiv_conv2d_2/GatherV2:output:05rot_equiv_conv2d_2/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  @*
paddingVALID*
strides
\
rot_equiv_conv2d_2/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 R{
rot_equiv_conv2d_2/Shape_1Shape!rot_equiv_pool2d_1/stack:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_conv2d_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€}
*rot_equiv_conv2d_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€t
*rot_equiv_conv2d_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ї
"rot_equiv_conv2d_2/strided_slice_1StridedSlice#rot_equiv_conv2d_2/Shape_1:output:01rot_equiv_conv2d_2/strided_slice_1/stack:output:03rot_equiv_conv2d_2/strided_slice_1/stack_1:output:03rot_equiv_conv2d_2/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_conv2d_2/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 RТ
rot_equiv_conv2d_2/sub_1Sub+rot_equiv_conv2d_2/strided_slice_1:output:0#rot_equiv_conv2d_2/sub_1/y:output:0*
T0	*
_output_shapes
: Щ
*rot_equiv_conv2d_2/clip_by_value_1/MinimumMinimum#rot_equiv_conv2d_2/Const_1:output:0rot_equiv_conv2d_2/sub_1:z:0*
T0	*
_output_shapes
: f
$rot_equiv_conv2d_2/clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ≠
"rot_equiv_conv2d_2/clip_by_value_1Maximum.rot_equiv_conv2d_2/clip_by_value_1/Minimum:z:0-rot_equiv_conv2d_2/clip_by_value_1/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_conv2d_2/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ю
rot_equiv_conv2d_2/GatherV2_1GatherV2!rot_equiv_pool2d_1/stack:output:0&rot_equiv_conv2d_2/clip_by_value_1:z:0+rot_equiv_conv2d_2/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€"" Ѓ
/rot_equiv_conv2d_2/convolution_1/ReadVariableOpReadVariableOp6rot_equiv_conv2d_2_convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0о
 rot_equiv_conv2d_2/convolution_1Conv2D&rot_equiv_conv2d_2/GatherV2_1:output:07rot_equiv_conv2d_2/convolution_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  @*
paddingVALID*
strides
\
rot_equiv_conv2d_2/Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R{
rot_equiv_conv2d_2/Shape_2Shape!rot_equiv_pool2d_1/stack:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_conv2d_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€}
*rot_equiv_conv2d_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€t
*rot_equiv_conv2d_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ї
"rot_equiv_conv2d_2/strided_slice_2StridedSlice#rot_equiv_conv2d_2/Shape_2:output:01rot_equiv_conv2d_2/strided_slice_2/stack:output:03rot_equiv_conv2d_2/strided_slice_2/stack_1:output:03rot_equiv_conv2d_2/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_conv2d_2/sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 RТ
rot_equiv_conv2d_2/sub_2Sub+rot_equiv_conv2d_2/strided_slice_2:output:0#rot_equiv_conv2d_2/sub_2/y:output:0*
T0	*
_output_shapes
: Щ
*rot_equiv_conv2d_2/clip_by_value_2/MinimumMinimum#rot_equiv_conv2d_2/Const_2:output:0rot_equiv_conv2d_2/sub_2:z:0*
T0	*
_output_shapes
: f
$rot_equiv_conv2d_2/clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ≠
"rot_equiv_conv2d_2/clip_by_value_2Maximum.rot_equiv_conv2d_2/clip_by_value_2/Minimum:z:0-rot_equiv_conv2d_2/clip_by_value_2/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_conv2d_2/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ю
rot_equiv_conv2d_2/GatherV2_2GatherV2!rot_equiv_pool2d_1/stack:output:0&rot_equiv_conv2d_2/clip_by_value_2:z:0+rot_equiv_conv2d_2/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€"" Ѓ
/rot_equiv_conv2d_2/convolution_2/ReadVariableOpReadVariableOp6rot_equiv_conv2d_2_convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0о
 rot_equiv_conv2d_2/convolution_2Conv2D&rot_equiv_conv2d_2/GatherV2_2:output:07rot_equiv_conv2d_2/convolution_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  @*
paddingVALID*
strides
\
rot_equiv_conv2d_2/Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 R{
rot_equiv_conv2d_2/Shape_3Shape!rot_equiv_pool2d_1/stack:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_conv2d_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€}
*rot_equiv_conv2d_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€t
*rot_equiv_conv2d_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ї
"rot_equiv_conv2d_2/strided_slice_3StridedSlice#rot_equiv_conv2d_2/Shape_3:output:01rot_equiv_conv2d_2/strided_slice_3/stack:output:03rot_equiv_conv2d_2/strided_slice_3/stack_1:output:03rot_equiv_conv2d_2/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_conv2d_2/sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 RТ
rot_equiv_conv2d_2/sub_3Sub+rot_equiv_conv2d_2/strided_slice_3:output:0#rot_equiv_conv2d_2/sub_3/y:output:0*
T0	*
_output_shapes
: Щ
*rot_equiv_conv2d_2/clip_by_value_3/MinimumMinimum#rot_equiv_conv2d_2/Const_3:output:0rot_equiv_conv2d_2/sub_3:z:0*
T0	*
_output_shapes
: f
$rot_equiv_conv2d_2/clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ≠
"rot_equiv_conv2d_2/clip_by_value_3Maximum.rot_equiv_conv2d_2/clip_by_value_3/Minimum:z:0-rot_equiv_conv2d_2/clip_by_value_3/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_conv2d_2/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ю
rot_equiv_conv2d_2/GatherV2_3GatherV2!rot_equiv_pool2d_1/stack:output:0&rot_equiv_conv2d_2/clip_by_value_3:z:0+rot_equiv_conv2d_2/GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€"" Ѓ
/rot_equiv_conv2d_2/convolution_3/ReadVariableOpReadVariableOp6rot_equiv_conv2d_2_convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0о
 rot_equiv_conv2d_2/convolution_3Conv2D&rot_equiv_conv2d_2/GatherV2_3:output:07rot_equiv_conv2d_2/convolution_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  @*
paddingVALID*
strides
¶
rot_equiv_conv2d_2/stackPack'rot_equiv_conv2d_2/convolution:output:0)rot_equiv_conv2d_2/convolution_1:output:0)rot_equiv_conv2d_2/convolution_2:output:0)rot_equiv_conv2d_2/convolution_3:output:0*
N*
T0*3
_output_shapes!
:€€€€€€€€€  @*
axisю€€€€€€€€А
rot_equiv_conv2d_2/ReluRelu!rot_equiv_conv2d_2/stack:output:0*
T0*3
_output_shapes!
:€€€€€€€€€  @Ш
)rot_equiv_conv2d_2/BiasAdd/ReadVariableOpReadVariableOp2rot_equiv_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0љ
rot_equiv_conv2d_2/BiasAddBiasAdd%rot_equiv_conv2d_2/Relu:activations:01rot_equiv_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€  @Z
rot_equiv_pool2d_2/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R {
rot_equiv_pool2d_2/ShapeShape#rot_equiv_conv2d_2/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	y
&rot_equiv_pool2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€{
(rot_equiv_pool2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€r
(rot_equiv_pool2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:∞
 rot_equiv_pool2d_2/strided_sliceStridedSlice!rot_equiv_pool2d_2/Shape:output:0/rot_equiv_pool2d_2/strided_slice/stack:output:01rot_equiv_pool2d_2/strided_slice/stack_1:output:01rot_equiv_pool2d_2/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskZ
rot_equiv_pool2d_2/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RМ
rot_equiv_pool2d_2/subSub)rot_equiv_pool2d_2/strided_slice:output:0!rot_equiv_pool2d_2/sub/y:output:0*
T0	*
_output_shapes
: У
(rot_equiv_pool2d_2/clip_by_value/MinimumMinimum!rot_equiv_pool2d_2/Const:output:0rot_equiv_pool2d_2/sub:z:0*
T0	*
_output_shapes
: d
"rot_equiv_pool2d_2/clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R І
 rot_equiv_pool2d_2/clip_by_valueMaximum,rot_equiv_pool2d_2/clip_by_value/Minimum:z:0+rot_equiv_pool2d_2/clip_by_value/y:output:0*
T0	*
_output_shapes
: k
 rot_equiv_pool2d_2/GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ъ
rot_equiv_pool2d_2/GatherV2GatherV2#rot_equiv_conv2d_2/BiasAdd:output:0$rot_equiv_pool2d_2/clip_by_value:z:0)rot_equiv_pool2d_2/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€  @»
*rot_equiv_pool2d_2/max_pooling2d_2/MaxPoolMaxPool$rot_equiv_pool2d_2/GatherV2:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
\
rot_equiv_pool2d_2/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 R}
rot_equiv_pool2d_2/Shape_1Shape#rot_equiv_conv2d_2/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_pool2d_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€}
*rot_equiv_pool2d_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€t
*rot_equiv_pool2d_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ї
"rot_equiv_pool2d_2/strided_slice_1StridedSlice#rot_equiv_pool2d_2/Shape_1:output:01rot_equiv_pool2d_2/strided_slice_1/stack:output:03rot_equiv_pool2d_2/strided_slice_1/stack_1:output:03rot_equiv_pool2d_2/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_pool2d_2/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 RТ
rot_equiv_pool2d_2/sub_1Sub+rot_equiv_pool2d_2/strided_slice_1:output:0#rot_equiv_pool2d_2/sub_1/y:output:0*
T0	*
_output_shapes
: Щ
*rot_equiv_pool2d_2/clip_by_value_1/MinimumMinimum#rot_equiv_pool2d_2/Const_1:output:0rot_equiv_pool2d_2/sub_1:z:0*
T0	*
_output_shapes
: f
$rot_equiv_pool2d_2/clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ≠
"rot_equiv_pool2d_2/clip_by_value_1Maximum.rot_equiv_pool2d_2/clip_by_value_1/Minimum:z:0-rot_equiv_pool2d_2/clip_by_value_1/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_pool2d_2/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€А
rot_equiv_pool2d_2/GatherV2_1GatherV2#rot_equiv_conv2d_2/BiasAdd:output:0&rot_equiv_pool2d_2/clip_by_value_1:z:0+rot_equiv_pool2d_2/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€  @ћ
,rot_equiv_pool2d_2/max_pooling2d_2/MaxPool_1MaxPool&rot_equiv_pool2d_2/GatherV2_1:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
\
rot_equiv_pool2d_2/Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R}
rot_equiv_pool2d_2/Shape_2Shape#rot_equiv_conv2d_2/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_pool2d_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€}
*rot_equiv_pool2d_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€t
*rot_equiv_pool2d_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ї
"rot_equiv_pool2d_2/strided_slice_2StridedSlice#rot_equiv_pool2d_2/Shape_2:output:01rot_equiv_pool2d_2/strided_slice_2/stack:output:03rot_equiv_pool2d_2/strided_slice_2/stack_1:output:03rot_equiv_pool2d_2/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_pool2d_2/sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 RТ
rot_equiv_pool2d_2/sub_2Sub+rot_equiv_pool2d_2/strided_slice_2:output:0#rot_equiv_pool2d_2/sub_2/y:output:0*
T0	*
_output_shapes
: Щ
*rot_equiv_pool2d_2/clip_by_value_2/MinimumMinimum#rot_equiv_pool2d_2/Const_2:output:0rot_equiv_pool2d_2/sub_2:z:0*
T0	*
_output_shapes
: f
$rot_equiv_pool2d_2/clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ≠
"rot_equiv_pool2d_2/clip_by_value_2Maximum.rot_equiv_pool2d_2/clip_by_value_2/Minimum:z:0-rot_equiv_pool2d_2/clip_by_value_2/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_pool2d_2/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€А
rot_equiv_pool2d_2/GatherV2_2GatherV2#rot_equiv_conv2d_2/BiasAdd:output:0&rot_equiv_pool2d_2/clip_by_value_2:z:0+rot_equiv_pool2d_2/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€  @ћ
,rot_equiv_pool2d_2/max_pooling2d_2/MaxPool_2MaxPool&rot_equiv_pool2d_2/GatherV2_2:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
\
rot_equiv_pool2d_2/Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 R}
rot_equiv_pool2d_2/Shape_3Shape#rot_equiv_conv2d_2/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_pool2d_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€}
*rot_equiv_pool2d_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€t
*rot_equiv_pool2d_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ї
"rot_equiv_pool2d_2/strided_slice_3StridedSlice#rot_equiv_pool2d_2/Shape_3:output:01rot_equiv_pool2d_2/strided_slice_3/stack:output:03rot_equiv_pool2d_2/strided_slice_3/stack_1:output:03rot_equiv_pool2d_2/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_pool2d_2/sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 RТ
rot_equiv_pool2d_2/sub_3Sub+rot_equiv_pool2d_2/strided_slice_3:output:0#rot_equiv_pool2d_2/sub_3/y:output:0*
T0	*
_output_shapes
: Щ
*rot_equiv_pool2d_2/clip_by_value_3/MinimumMinimum#rot_equiv_pool2d_2/Const_3:output:0rot_equiv_pool2d_2/sub_3:z:0*
T0	*
_output_shapes
: f
$rot_equiv_pool2d_2/clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ≠
"rot_equiv_pool2d_2/clip_by_value_3Maximum.rot_equiv_pool2d_2/clip_by_value_3/Minimum:z:0-rot_equiv_pool2d_2/clip_by_value_3/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_pool2d_2/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€А
rot_equiv_pool2d_2/GatherV2_3GatherV2#rot_equiv_conv2d_2/BiasAdd:output:0&rot_equiv_pool2d_2/clip_by_value_3:z:0+rot_equiv_pool2d_2/GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€  @ћ
,rot_equiv_pool2d_2/max_pooling2d_2/MaxPool_3MaxPool&rot_equiv_pool2d_2/GatherV2_3:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
÷
rot_equiv_pool2d_2/stackPack3rot_equiv_pool2d_2/max_pooling2d_2/MaxPool:output:05rot_equiv_pool2d_2/max_pooling2d_2/MaxPool_1:output:05rot_equiv_pool2d_2/max_pooling2d_2/MaxPool_2:output:05rot_equiv_pool2d_2/max_pooling2d_2/MaxPool_3:output:0*
N*
T0*3
_output_shapes!
:€€€€€€€€€@*
axisю€€€€€€€€Z
rot_equiv_conv2d_3/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R y
rot_equiv_conv2d_3/ShapeShape!rot_equiv_pool2d_2/stack:output:0*
T0*
_output_shapes
:*
out_type0	y
&rot_equiv_conv2d_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€{
(rot_equiv_conv2d_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€r
(rot_equiv_conv2d_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:∞
 rot_equiv_conv2d_3/strided_sliceStridedSlice!rot_equiv_conv2d_3/Shape:output:0/rot_equiv_conv2d_3/strided_slice/stack:output:01rot_equiv_conv2d_3/strided_slice/stack_1:output:01rot_equiv_conv2d_3/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskZ
rot_equiv_conv2d_3/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RМ
rot_equiv_conv2d_3/subSub)rot_equiv_conv2d_3/strided_slice:output:0!rot_equiv_conv2d_3/sub/y:output:0*
T0	*
_output_shapes
: У
(rot_equiv_conv2d_3/clip_by_value/MinimumMinimum!rot_equiv_conv2d_3/Const:output:0rot_equiv_conv2d_3/sub:z:0*
T0	*
_output_shapes
: d
"rot_equiv_conv2d_3/clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R І
 rot_equiv_conv2d_3/clip_by_valueMaximum,rot_equiv_conv2d_3/clip_by_value/Minimum:z:0+rot_equiv_conv2d_3/clip_by_value/y:output:0*
T0	*
_output_shapes
: k
 rot_equiv_conv2d_3/GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ш
rot_equiv_conv2d_3/GatherV2GatherV2!rot_equiv_pool2d_2/stack:output:0$rot_equiv_conv2d_3/clip_by_value:z:0)rot_equiv_conv2d_3/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@ђ
-rot_equiv_conv2d_3/convolution/ReadVariableOpReadVariableOp6rot_equiv_conv2d_3_convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype0и
rot_equiv_conv2d_3/convolutionConv2D$rot_equiv_conv2d_3/GatherV2:output:05rot_equiv_conv2d_3/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingVALID*
strides
\
rot_equiv_conv2d_3/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 R{
rot_equiv_conv2d_3/Shape_1Shape!rot_equiv_pool2d_2/stack:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_conv2d_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€}
*rot_equiv_conv2d_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€t
*rot_equiv_conv2d_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ї
"rot_equiv_conv2d_3/strided_slice_1StridedSlice#rot_equiv_conv2d_3/Shape_1:output:01rot_equiv_conv2d_3/strided_slice_1/stack:output:03rot_equiv_conv2d_3/strided_slice_1/stack_1:output:03rot_equiv_conv2d_3/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_conv2d_3/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 RТ
rot_equiv_conv2d_3/sub_1Sub+rot_equiv_conv2d_3/strided_slice_1:output:0#rot_equiv_conv2d_3/sub_1/y:output:0*
T0	*
_output_shapes
: Щ
*rot_equiv_conv2d_3/clip_by_value_1/MinimumMinimum#rot_equiv_conv2d_3/Const_1:output:0rot_equiv_conv2d_3/sub_1:z:0*
T0	*
_output_shapes
: f
$rot_equiv_conv2d_3/clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ≠
"rot_equiv_conv2d_3/clip_by_value_1Maximum.rot_equiv_conv2d_3/clip_by_value_1/Minimum:z:0-rot_equiv_conv2d_3/clip_by_value_1/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_conv2d_3/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ю
rot_equiv_conv2d_3/GatherV2_1GatherV2!rot_equiv_pool2d_2/stack:output:0&rot_equiv_conv2d_3/clip_by_value_1:z:0+rot_equiv_conv2d_3/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@Ѓ
/rot_equiv_conv2d_3/convolution_1/ReadVariableOpReadVariableOp6rot_equiv_conv2d_3_convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype0о
 rot_equiv_conv2d_3/convolution_1Conv2D&rot_equiv_conv2d_3/GatherV2_1:output:07rot_equiv_conv2d_3/convolution_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingVALID*
strides
\
rot_equiv_conv2d_3/Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R{
rot_equiv_conv2d_3/Shape_2Shape!rot_equiv_pool2d_2/stack:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_conv2d_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€}
*rot_equiv_conv2d_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€t
*rot_equiv_conv2d_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ї
"rot_equiv_conv2d_3/strided_slice_2StridedSlice#rot_equiv_conv2d_3/Shape_2:output:01rot_equiv_conv2d_3/strided_slice_2/stack:output:03rot_equiv_conv2d_3/strided_slice_2/stack_1:output:03rot_equiv_conv2d_3/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_conv2d_3/sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 RТ
rot_equiv_conv2d_3/sub_2Sub+rot_equiv_conv2d_3/strided_slice_2:output:0#rot_equiv_conv2d_3/sub_2/y:output:0*
T0	*
_output_shapes
: Щ
*rot_equiv_conv2d_3/clip_by_value_2/MinimumMinimum#rot_equiv_conv2d_3/Const_2:output:0rot_equiv_conv2d_3/sub_2:z:0*
T0	*
_output_shapes
: f
$rot_equiv_conv2d_3/clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ≠
"rot_equiv_conv2d_3/clip_by_value_2Maximum.rot_equiv_conv2d_3/clip_by_value_2/Minimum:z:0-rot_equiv_conv2d_3/clip_by_value_2/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_conv2d_3/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ю
rot_equiv_conv2d_3/GatherV2_2GatherV2!rot_equiv_pool2d_2/stack:output:0&rot_equiv_conv2d_3/clip_by_value_2:z:0+rot_equiv_conv2d_3/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@Ѓ
/rot_equiv_conv2d_3/convolution_2/ReadVariableOpReadVariableOp6rot_equiv_conv2d_3_convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype0о
 rot_equiv_conv2d_3/convolution_2Conv2D&rot_equiv_conv2d_3/GatherV2_2:output:07rot_equiv_conv2d_3/convolution_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingVALID*
strides
\
rot_equiv_conv2d_3/Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 R{
rot_equiv_conv2d_3/Shape_3Shape!rot_equiv_pool2d_2/stack:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_conv2d_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€}
*rot_equiv_conv2d_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€t
*rot_equiv_conv2d_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ї
"rot_equiv_conv2d_3/strided_slice_3StridedSlice#rot_equiv_conv2d_3/Shape_3:output:01rot_equiv_conv2d_3/strided_slice_3/stack:output:03rot_equiv_conv2d_3/strided_slice_3/stack_1:output:03rot_equiv_conv2d_3/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_conv2d_3/sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 RТ
rot_equiv_conv2d_3/sub_3Sub+rot_equiv_conv2d_3/strided_slice_3:output:0#rot_equiv_conv2d_3/sub_3/y:output:0*
T0	*
_output_shapes
: Щ
*rot_equiv_conv2d_3/clip_by_value_3/MinimumMinimum#rot_equiv_conv2d_3/Const_3:output:0rot_equiv_conv2d_3/sub_3:z:0*
T0	*
_output_shapes
: f
$rot_equiv_conv2d_3/clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ≠
"rot_equiv_conv2d_3/clip_by_value_3Maximum.rot_equiv_conv2d_3/clip_by_value_3/Minimum:z:0-rot_equiv_conv2d_3/clip_by_value_3/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_conv2d_3/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ю
rot_equiv_conv2d_3/GatherV2_3GatherV2!rot_equiv_pool2d_2/stack:output:0&rot_equiv_conv2d_3/clip_by_value_3:z:0+rot_equiv_conv2d_3/GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@Ѓ
/rot_equiv_conv2d_3/convolution_3/ReadVariableOpReadVariableOp6rot_equiv_conv2d_3_convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype0о
 rot_equiv_conv2d_3/convolution_3Conv2D&rot_equiv_conv2d_3/GatherV2_3:output:07rot_equiv_conv2d_3/convolution_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingVALID*
strides
¶
rot_equiv_conv2d_3/stackPack'rot_equiv_conv2d_3/convolution:output:0)rot_equiv_conv2d_3/convolution_1:output:0)rot_equiv_conv2d_3/convolution_2:output:0)rot_equiv_conv2d_3/convolution_3:output:0*
N*
T0*3
_output_shapes!
:€€€€€€€€€@*
axisю€€€€€€€€А
rot_equiv_conv2d_3/ReluRelu!rot_equiv_conv2d_3/stack:output:0*
T0*3
_output_shapes!
:€€€€€€€€€@Ш
)rot_equiv_conv2d_3/BiasAdd/ReadVariableOpReadVariableOp2rot_equiv_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0љ
rot_equiv_conv2d_3/BiasAddBiasAdd%rot_equiv_conv2d_3/Relu:activations:01rot_equiv_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€@Z
rot_equiv_pool2d_3/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R {
rot_equiv_pool2d_3/ShapeShape#rot_equiv_conv2d_3/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	y
&rot_equiv_pool2d_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€{
(rot_equiv_pool2d_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€r
(rot_equiv_pool2d_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:∞
 rot_equiv_pool2d_3/strided_sliceStridedSlice!rot_equiv_pool2d_3/Shape:output:0/rot_equiv_pool2d_3/strided_slice/stack:output:01rot_equiv_pool2d_3/strided_slice/stack_1:output:01rot_equiv_pool2d_3/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskZ
rot_equiv_pool2d_3/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RМ
rot_equiv_pool2d_3/subSub)rot_equiv_pool2d_3/strided_slice:output:0!rot_equiv_pool2d_3/sub/y:output:0*
T0	*
_output_shapes
: У
(rot_equiv_pool2d_3/clip_by_value/MinimumMinimum!rot_equiv_pool2d_3/Const:output:0rot_equiv_pool2d_3/sub:z:0*
T0	*
_output_shapes
: d
"rot_equiv_pool2d_3/clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R І
 rot_equiv_pool2d_3/clip_by_valueMaximum,rot_equiv_pool2d_3/clip_by_value/Minimum:z:0+rot_equiv_pool2d_3/clip_by_value/y:output:0*
T0	*
_output_shapes
: k
 rot_equiv_pool2d_3/GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ъ
rot_equiv_pool2d_3/GatherV2GatherV2#rot_equiv_conv2d_3/BiasAdd:output:0$rot_equiv_pool2d_3/clip_by_value:z:0)rot_equiv_pool2d_3/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@»
*rot_equiv_pool2d_3/max_pooling2d_3/MaxPoolMaxPool$rot_equiv_pool2d_3/GatherV2:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
\
rot_equiv_pool2d_3/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 R}
rot_equiv_pool2d_3/Shape_1Shape#rot_equiv_conv2d_3/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_pool2d_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€}
*rot_equiv_pool2d_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€t
*rot_equiv_pool2d_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ї
"rot_equiv_pool2d_3/strided_slice_1StridedSlice#rot_equiv_pool2d_3/Shape_1:output:01rot_equiv_pool2d_3/strided_slice_1/stack:output:03rot_equiv_pool2d_3/strided_slice_1/stack_1:output:03rot_equiv_pool2d_3/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_pool2d_3/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 RТ
rot_equiv_pool2d_3/sub_1Sub+rot_equiv_pool2d_3/strided_slice_1:output:0#rot_equiv_pool2d_3/sub_1/y:output:0*
T0	*
_output_shapes
: Щ
*rot_equiv_pool2d_3/clip_by_value_1/MinimumMinimum#rot_equiv_pool2d_3/Const_1:output:0rot_equiv_pool2d_3/sub_1:z:0*
T0	*
_output_shapes
: f
$rot_equiv_pool2d_3/clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ≠
"rot_equiv_pool2d_3/clip_by_value_1Maximum.rot_equiv_pool2d_3/clip_by_value_1/Minimum:z:0-rot_equiv_pool2d_3/clip_by_value_1/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_pool2d_3/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€А
rot_equiv_pool2d_3/GatherV2_1GatherV2#rot_equiv_conv2d_3/BiasAdd:output:0&rot_equiv_pool2d_3/clip_by_value_1:z:0+rot_equiv_pool2d_3/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@ћ
,rot_equiv_pool2d_3/max_pooling2d_3/MaxPool_1MaxPool&rot_equiv_pool2d_3/GatherV2_1:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
\
rot_equiv_pool2d_3/Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R}
rot_equiv_pool2d_3/Shape_2Shape#rot_equiv_conv2d_3/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_pool2d_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€}
*rot_equiv_pool2d_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€t
*rot_equiv_pool2d_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ї
"rot_equiv_pool2d_3/strided_slice_2StridedSlice#rot_equiv_pool2d_3/Shape_2:output:01rot_equiv_pool2d_3/strided_slice_2/stack:output:03rot_equiv_pool2d_3/strided_slice_2/stack_1:output:03rot_equiv_pool2d_3/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_pool2d_3/sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 RТ
rot_equiv_pool2d_3/sub_2Sub+rot_equiv_pool2d_3/strided_slice_2:output:0#rot_equiv_pool2d_3/sub_2/y:output:0*
T0	*
_output_shapes
: Щ
*rot_equiv_pool2d_3/clip_by_value_2/MinimumMinimum#rot_equiv_pool2d_3/Const_2:output:0rot_equiv_pool2d_3/sub_2:z:0*
T0	*
_output_shapes
: f
$rot_equiv_pool2d_3/clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ≠
"rot_equiv_pool2d_3/clip_by_value_2Maximum.rot_equiv_pool2d_3/clip_by_value_2/Minimum:z:0-rot_equiv_pool2d_3/clip_by_value_2/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_pool2d_3/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€А
rot_equiv_pool2d_3/GatherV2_2GatherV2#rot_equiv_conv2d_3/BiasAdd:output:0&rot_equiv_pool2d_3/clip_by_value_2:z:0+rot_equiv_pool2d_3/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@ћ
,rot_equiv_pool2d_3/max_pooling2d_3/MaxPool_2MaxPool&rot_equiv_pool2d_3/GatherV2_2:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
\
rot_equiv_pool2d_3/Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 R}
rot_equiv_pool2d_3/Shape_3Shape#rot_equiv_conv2d_3/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_pool2d_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€}
*rot_equiv_pool2d_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€t
*rot_equiv_pool2d_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ї
"rot_equiv_pool2d_3/strided_slice_3StridedSlice#rot_equiv_pool2d_3/Shape_3:output:01rot_equiv_pool2d_3/strided_slice_3/stack:output:03rot_equiv_pool2d_3/strided_slice_3/stack_1:output:03rot_equiv_pool2d_3/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_pool2d_3/sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 RТ
rot_equiv_pool2d_3/sub_3Sub+rot_equiv_pool2d_3/strided_slice_3:output:0#rot_equiv_pool2d_3/sub_3/y:output:0*
T0	*
_output_shapes
: Щ
*rot_equiv_pool2d_3/clip_by_value_3/MinimumMinimum#rot_equiv_pool2d_3/Const_3:output:0rot_equiv_pool2d_3/sub_3:z:0*
T0	*
_output_shapes
: f
$rot_equiv_pool2d_3/clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ≠
"rot_equiv_pool2d_3/clip_by_value_3Maximum.rot_equiv_pool2d_3/clip_by_value_3/Minimum:z:0-rot_equiv_pool2d_3/clip_by_value_3/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_pool2d_3/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€А
rot_equiv_pool2d_3/GatherV2_3GatherV2#rot_equiv_conv2d_3/BiasAdd:output:0&rot_equiv_pool2d_3/clip_by_value_3:z:0+rot_equiv_pool2d_3/GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@ћ
,rot_equiv_pool2d_3/max_pooling2d_3/MaxPool_3MaxPool&rot_equiv_pool2d_3/GatherV2_3:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
÷
rot_equiv_pool2d_3/stackPack3rot_equiv_pool2d_3/max_pooling2d_3/MaxPool:output:05rot_equiv_pool2d_3/max_pooling2d_3/MaxPool_1:output:05rot_equiv_pool2d_3/max_pooling2d_3/MaxPool_2:output:05rot_equiv_pool2d_3/max_pooling2d_3/MaxPool_3:output:0*
N*
T0*3
_output_shapes!
:€€€€€€€€€@*
axisю€€€€€€€€Z
rot_equiv_conv2d_4/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R y
rot_equiv_conv2d_4/ShapeShape!rot_equiv_pool2d_3/stack:output:0*
T0*
_output_shapes
:*
out_type0	y
&rot_equiv_conv2d_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€{
(rot_equiv_conv2d_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€r
(rot_equiv_conv2d_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:∞
 rot_equiv_conv2d_4/strided_sliceStridedSlice!rot_equiv_conv2d_4/Shape:output:0/rot_equiv_conv2d_4/strided_slice/stack:output:01rot_equiv_conv2d_4/strided_slice/stack_1:output:01rot_equiv_conv2d_4/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskZ
rot_equiv_conv2d_4/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RМ
rot_equiv_conv2d_4/subSub)rot_equiv_conv2d_4/strided_slice:output:0!rot_equiv_conv2d_4/sub/y:output:0*
T0	*
_output_shapes
: У
(rot_equiv_conv2d_4/clip_by_value/MinimumMinimum!rot_equiv_conv2d_4/Const:output:0rot_equiv_conv2d_4/sub:z:0*
T0	*
_output_shapes
: d
"rot_equiv_conv2d_4/clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R І
 rot_equiv_conv2d_4/clip_by_valueMaximum,rot_equiv_conv2d_4/clip_by_value/Minimum:z:0+rot_equiv_conv2d_4/clip_by_value/y:output:0*
T0	*
_output_shapes
: k
 rot_equiv_conv2d_4/GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ш
rot_equiv_conv2d_4/GatherV2GatherV2!rot_equiv_pool2d_3/stack:output:0$rot_equiv_conv2d_4/clip_by_value:z:0)rot_equiv_conv2d_4/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@≠
-rot_equiv_conv2d_4/convolution/ReadVariableOpReadVariableOp6rot_equiv_conv2d_4_convolution_readvariableop_resource*'
_output_shapes
:@А*
dtype0й
rot_equiv_conv2d_4/convolutionConv2D$rot_equiv_conv2d_4/GatherV2:output:05rot_equiv_conv2d_4/convolution/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingVALID*
strides
\
rot_equiv_conv2d_4/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 R{
rot_equiv_conv2d_4/Shape_1Shape!rot_equiv_pool2d_3/stack:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_conv2d_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€}
*rot_equiv_conv2d_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€t
*rot_equiv_conv2d_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ї
"rot_equiv_conv2d_4/strided_slice_1StridedSlice#rot_equiv_conv2d_4/Shape_1:output:01rot_equiv_conv2d_4/strided_slice_1/stack:output:03rot_equiv_conv2d_4/strided_slice_1/stack_1:output:03rot_equiv_conv2d_4/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_conv2d_4/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 RТ
rot_equiv_conv2d_4/sub_1Sub+rot_equiv_conv2d_4/strided_slice_1:output:0#rot_equiv_conv2d_4/sub_1/y:output:0*
T0	*
_output_shapes
: Щ
*rot_equiv_conv2d_4/clip_by_value_1/MinimumMinimum#rot_equiv_conv2d_4/Const_1:output:0rot_equiv_conv2d_4/sub_1:z:0*
T0	*
_output_shapes
: f
$rot_equiv_conv2d_4/clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ≠
"rot_equiv_conv2d_4/clip_by_value_1Maximum.rot_equiv_conv2d_4/clip_by_value_1/Minimum:z:0-rot_equiv_conv2d_4/clip_by_value_1/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_conv2d_4/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ю
rot_equiv_conv2d_4/GatherV2_1GatherV2!rot_equiv_pool2d_3/stack:output:0&rot_equiv_conv2d_4/clip_by_value_1:z:0+rot_equiv_conv2d_4/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@ѓ
/rot_equiv_conv2d_4/convolution_1/ReadVariableOpReadVariableOp6rot_equiv_conv2d_4_convolution_readvariableop_resource*'
_output_shapes
:@А*
dtype0п
 rot_equiv_conv2d_4/convolution_1Conv2D&rot_equiv_conv2d_4/GatherV2_1:output:07rot_equiv_conv2d_4/convolution_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingVALID*
strides
\
rot_equiv_conv2d_4/Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R{
rot_equiv_conv2d_4/Shape_2Shape!rot_equiv_pool2d_3/stack:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_conv2d_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€}
*rot_equiv_conv2d_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€t
*rot_equiv_conv2d_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ї
"rot_equiv_conv2d_4/strided_slice_2StridedSlice#rot_equiv_conv2d_4/Shape_2:output:01rot_equiv_conv2d_4/strided_slice_2/stack:output:03rot_equiv_conv2d_4/strided_slice_2/stack_1:output:03rot_equiv_conv2d_4/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_conv2d_4/sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 RТ
rot_equiv_conv2d_4/sub_2Sub+rot_equiv_conv2d_4/strided_slice_2:output:0#rot_equiv_conv2d_4/sub_2/y:output:0*
T0	*
_output_shapes
: Щ
*rot_equiv_conv2d_4/clip_by_value_2/MinimumMinimum#rot_equiv_conv2d_4/Const_2:output:0rot_equiv_conv2d_4/sub_2:z:0*
T0	*
_output_shapes
: f
$rot_equiv_conv2d_4/clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ≠
"rot_equiv_conv2d_4/clip_by_value_2Maximum.rot_equiv_conv2d_4/clip_by_value_2/Minimum:z:0-rot_equiv_conv2d_4/clip_by_value_2/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_conv2d_4/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ю
rot_equiv_conv2d_4/GatherV2_2GatherV2!rot_equiv_pool2d_3/stack:output:0&rot_equiv_conv2d_4/clip_by_value_2:z:0+rot_equiv_conv2d_4/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@ѓ
/rot_equiv_conv2d_4/convolution_2/ReadVariableOpReadVariableOp6rot_equiv_conv2d_4_convolution_readvariableop_resource*'
_output_shapes
:@А*
dtype0п
 rot_equiv_conv2d_4/convolution_2Conv2D&rot_equiv_conv2d_4/GatherV2_2:output:07rot_equiv_conv2d_4/convolution_2/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingVALID*
strides
\
rot_equiv_conv2d_4/Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 R{
rot_equiv_conv2d_4/Shape_3Shape!rot_equiv_pool2d_3/stack:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_conv2d_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€}
*rot_equiv_conv2d_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€t
*rot_equiv_conv2d_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ї
"rot_equiv_conv2d_4/strided_slice_3StridedSlice#rot_equiv_conv2d_4/Shape_3:output:01rot_equiv_conv2d_4/strided_slice_3/stack:output:03rot_equiv_conv2d_4/strided_slice_3/stack_1:output:03rot_equiv_conv2d_4/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_conv2d_4/sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 RТ
rot_equiv_conv2d_4/sub_3Sub+rot_equiv_conv2d_4/strided_slice_3:output:0#rot_equiv_conv2d_4/sub_3/y:output:0*
T0	*
_output_shapes
: Щ
*rot_equiv_conv2d_4/clip_by_value_3/MinimumMinimum#rot_equiv_conv2d_4/Const_3:output:0rot_equiv_conv2d_4/sub_3:z:0*
T0	*
_output_shapes
: f
$rot_equiv_conv2d_4/clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ≠
"rot_equiv_conv2d_4/clip_by_value_3Maximum.rot_equiv_conv2d_4/clip_by_value_3/Minimum:z:0-rot_equiv_conv2d_4/clip_by_value_3/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_conv2d_4/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€ю
rot_equiv_conv2d_4/GatherV2_3GatherV2!rot_equiv_pool2d_3/stack:output:0&rot_equiv_conv2d_4/clip_by_value_3:z:0+rot_equiv_conv2d_4/GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€@ѓ
/rot_equiv_conv2d_4/convolution_3/ReadVariableOpReadVariableOp6rot_equiv_conv2d_4_convolution_readvariableop_resource*'
_output_shapes
:@А*
dtype0п
 rot_equiv_conv2d_4/convolution_3Conv2D&rot_equiv_conv2d_4/GatherV2_3:output:07rot_equiv_conv2d_4/convolution_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingVALID*
strides
І
rot_equiv_conv2d_4/stackPack'rot_equiv_conv2d_4/convolution:output:0)rot_equiv_conv2d_4/convolution_1:output:0)rot_equiv_conv2d_4/convolution_2:output:0)rot_equiv_conv2d_4/convolution_3:output:0*
N*
T0*4
_output_shapes"
 :€€€€€€€€€А*
axisю€€€€€€€€Б
rot_equiv_conv2d_4/ReluRelu!rot_equiv_conv2d_4/stack:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€АЩ
)rot_equiv_conv2d_4/BiasAdd/ReadVariableOpReadVariableOp2rot_equiv_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Њ
rot_equiv_conv2d_4/BiasAddBiasAdd%rot_equiv_conv2d_4/Relu:activations:01rot_equiv_conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аm
"rot_inv_pool/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€§
rot_inv_pool/MaxMax#rot_equiv_conv2d_4/BiasAdd:output:0+rot_inv_pool/Max/reduction_indices:output:0*
T0*0
_output_shapes
:€€€€€€€€€А^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€А  А
flatten/ReshapeReshaperot_inv_pool/Max:output:0flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€АБ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	А *
dtype0З
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0И
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ \

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ Д
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Л
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€В
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€g
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ќ

NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp(^rot_equiv_conv2d/BiasAdd/ReadVariableOp ^rot_equiv_conv2d/ReadVariableOp"^rot_equiv_conv2d/ReadVariableOp_1"^rot_equiv_conv2d/ReadVariableOp_2,^rot_equiv_conv2d/convolution/ReadVariableOp*^rot_equiv_conv2d_1/BiasAdd/ReadVariableOp.^rot_equiv_conv2d_1/convolution/ReadVariableOp0^rot_equiv_conv2d_1/convolution_1/ReadVariableOp0^rot_equiv_conv2d_1/convolution_2/ReadVariableOp0^rot_equiv_conv2d_1/convolution_3/ReadVariableOp*^rot_equiv_conv2d_2/BiasAdd/ReadVariableOp.^rot_equiv_conv2d_2/convolution/ReadVariableOp0^rot_equiv_conv2d_2/convolution_1/ReadVariableOp0^rot_equiv_conv2d_2/convolution_2/ReadVariableOp0^rot_equiv_conv2d_2/convolution_3/ReadVariableOp*^rot_equiv_conv2d_3/BiasAdd/ReadVariableOp.^rot_equiv_conv2d_3/convolution/ReadVariableOp0^rot_equiv_conv2d_3/convolution_1/ReadVariableOp0^rot_equiv_conv2d_3/convolution_2/ReadVariableOp0^rot_equiv_conv2d_3/convolution_3/ReadVariableOp*^rot_equiv_conv2d_4/BiasAdd/ReadVariableOp.^rot_equiv_conv2d_4/convolution/ReadVariableOp0^rot_equiv_conv2d_4/convolution_1/ReadVariableOp0^rot_equiv_conv2d_4/convolution_2/ReadVariableOp0^rot_equiv_conv2d_4/convolution_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:€€€€€€€€€РР: : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2R
'rot_equiv_conv2d/BiasAdd/ReadVariableOp'rot_equiv_conv2d/BiasAdd/ReadVariableOp2B
rot_equiv_conv2d/ReadVariableOprot_equiv_conv2d/ReadVariableOp2F
!rot_equiv_conv2d/ReadVariableOp_1!rot_equiv_conv2d/ReadVariableOp_12F
!rot_equiv_conv2d/ReadVariableOp_2!rot_equiv_conv2d/ReadVariableOp_22Z
+rot_equiv_conv2d/convolution/ReadVariableOp+rot_equiv_conv2d/convolution/ReadVariableOp2V
)rot_equiv_conv2d_1/BiasAdd/ReadVariableOp)rot_equiv_conv2d_1/BiasAdd/ReadVariableOp2^
-rot_equiv_conv2d_1/convolution/ReadVariableOp-rot_equiv_conv2d_1/convolution/ReadVariableOp2b
/rot_equiv_conv2d_1/convolution_1/ReadVariableOp/rot_equiv_conv2d_1/convolution_1/ReadVariableOp2b
/rot_equiv_conv2d_1/convolution_2/ReadVariableOp/rot_equiv_conv2d_1/convolution_2/ReadVariableOp2b
/rot_equiv_conv2d_1/convolution_3/ReadVariableOp/rot_equiv_conv2d_1/convolution_3/ReadVariableOp2V
)rot_equiv_conv2d_2/BiasAdd/ReadVariableOp)rot_equiv_conv2d_2/BiasAdd/ReadVariableOp2^
-rot_equiv_conv2d_2/convolution/ReadVariableOp-rot_equiv_conv2d_2/convolution/ReadVariableOp2b
/rot_equiv_conv2d_2/convolution_1/ReadVariableOp/rot_equiv_conv2d_2/convolution_1/ReadVariableOp2b
/rot_equiv_conv2d_2/convolution_2/ReadVariableOp/rot_equiv_conv2d_2/convolution_2/ReadVariableOp2b
/rot_equiv_conv2d_2/convolution_3/ReadVariableOp/rot_equiv_conv2d_2/convolution_3/ReadVariableOp2V
)rot_equiv_conv2d_3/BiasAdd/ReadVariableOp)rot_equiv_conv2d_3/BiasAdd/ReadVariableOp2^
-rot_equiv_conv2d_3/convolution/ReadVariableOp-rot_equiv_conv2d_3/convolution/ReadVariableOp2b
/rot_equiv_conv2d_3/convolution_1/ReadVariableOp/rot_equiv_conv2d_3/convolution_1/ReadVariableOp2b
/rot_equiv_conv2d_3/convolution_2/ReadVariableOp/rot_equiv_conv2d_3/convolution_2/ReadVariableOp2b
/rot_equiv_conv2d_3/convolution_3/ReadVariableOp/rot_equiv_conv2d_3/convolution_3/ReadVariableOp2V
)rot_equiv_conv2d_4/BiasAdd/ReadVariableOp)rot_equiv_conv2d_4/BiasAdd/ReadVariableOp2^
-rot_equiv_conv2d_4/convolution/ReadVariableOp-rot_equiv_conv2d_4/convolution/ReadVariableOp2b
/rot_equiv_conv2d_4/convolution_1/ReadVariableOp/rot_equiv_conv2d_4/convolution_1/ReadVariableOp2b
/rot_equiv_conv2d_4/convolution_2/ReadVariableOp/rot_equiv_conv2d_4/convolution_2/ReadVariableOp2b
/rot_equiv_conv2d_4/convolution_3/ReadVariableOp/rot_equiv_conv2d_4/convolution_3/ReadVariableOp:Y U
1
_output_shapes
:€€€€€€€€€РР
 
_user_specified_nameinputs
еC
о
N__inference_rot_equiv_conv2d_2_layer_call_and_return_conditional_losses_627578

inputs=
#convolution_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐconvolution/ReadVariableOpҐconvolution_1/ReadVariableOpҐconvolution_2/ReadVariableOpҐconvolution_3/ReadVariableOpG
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R K
ShapeShapeinputs*
T0*
_output_shapes
:*
out_type0	f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskG
sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RS
subSubstrided_slice:output:0sub/y:output:0*
T0	*
_output_shapes
: Z
clip_by_value/MinimumMinimumConst:output:0sub:z:0*
T0	*
_output_shapes
: Q
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R n
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*
_output_shapes
: X
GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€§
GatherV2GatherV2inputsclip_by_value:z:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€"" Ж
convolution/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0ѓ
convolutionConv2DGatherV2:output:0"convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  @*
paddingVALID*
strides
I
Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_1Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_1Substrided_slice_1:output:0sub_1/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_1/MinimumMinimumConst_1:output:0	sub_1:z:0*
T0	*
_output_shapes
: S
clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€™

GatherV2_1GatherV2inputsclip_by_value_1:z:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€"" И
convolution_1/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0µ
convolution_1Conv2DGatherV2_1:output:0$convolution_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  @*
paddingVALID*
strides
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_2Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_2StridedSliceShape_2:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_2Substrided_slice_2:output:0sub_2/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_2/MinimumMinimumConst_2:output:0	sub_2:z:0*
T0	*
_output_shapes
: S
clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€™

GatherV2_2GatherV2inputsclip_by_value_2:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€"" И
convolution_2/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0µ
convolution_2Conv2DGatherV2_2:output:0$convolution_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  @*
paddingVALID*
strides
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_3Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€j
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_3StridedSliceShape_3:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_3Substrided_slice_3:output:0sub_3/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_3/MinimumMinimumConst_3:output:0	sub_3:z:0*
T0	*
_output_shapes
: S
clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_3Maximumclip_by_value_3/Minimum:z:0clip_by_value_3/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€™

GatherV2_3GatherV2inputsclip_by_value_3:z:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€"" И
convolution_3/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0µ
convolution_3Conv2DGatherV2_3:output:0$convolution_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  @*
paddingVALID*
strides
«
stackPackconvolution:output:0convolution_1:output:0convolution_2:output:0convolution_3:output:0*
N*
T0*3
_output_shapes!
:€€€€€€€€€  @*
axisю€€€€€€€€Z
ReluRelustack:output:0*
T0*3
_output_shapes!
:€€€€€€€€€  @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Д
BiasAddBiasAddRelu:activations:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€  @k
IdentityIdentityBiasAdd:output:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€  @ў
NoOpNoOp^BiasAdd/ReadVariableOp^convolution/ReadVariableOp^convolution_1/ReadVariableOp^convolution_2/ReadVariableOp^convolution_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€"" : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp28
convolution/ReadVariableOpconvolution/ReadVariableOp2<
convolution_1/ReadVariableOpconvolution_1/ReadVariableOp2<
convolution_2/ReadVariableOpconvolution_2/ReadVariableOp2<
convolution_3/ReadVariableOpconvolution_3/ReadVariableOp:[ W
3
_output_shapes!
:€€€€€€€€€"" 
 
_user_specified_nameinputs
ї
L
0__inference_max_pooling2d_2_layer_call_fn_627954

inputs
identity№
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_624833Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
√
Х
(__inference_dense_1_layer_call_fn_627919

inputs
unknown: 
	unknown_0:
identityИҐStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_625539o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
¬
Ф
&__inference_dense_layer_call_fn_627899

inputs
unknown:	А 
	unknown_0: 
identityИҐStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_625523o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
«
_
C__inference_flatten_layer_call_and_return_conditional_losses_627890

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€А  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€АY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
т
d
H__inference_rot_inv_pool_layer_call_and_return_conditional_losses_627879

inputs
identity`
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€m
MaxMaxinputsMax/reduction_indices:output:0*
T0*0
_output_shapes
:€€€€€€€€€А]
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€А:\ X
4
_output_shapes"
 :€€€€€€€€€А
 
_user_specified_nameinputs
С
e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_624809

inputs
identityҐ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ь

у
A__inference_dense_layer_call_and_return_conditional_losses_627910

inputs1
matmul_readvariableop_resource:	А -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
ї
L
0__inference_max_pooling2d_3_layer_call_fn_627964

inputs
identity№
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_624845Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
С
e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_627939

inputs
identityҐ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
т
d
H__inference_rot_inv_pool_layer_call_and_return_conditional_losses_625502

inputs
identity`
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€m
MaxMaxinputsMax/reduction_indices:output:0*
T0*0
_output_shapes
:€€€€€€€€€А]
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€А:\ X
4
_output_shapes"
 :€€€€€€€€€А
 
_user_specified_nameinputs
∆6
j
N__inference_rot_equiv_pool2d_2_layer_call_and_return_conditional_losses_627644

inputs
identityG
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R K
ShapeShapeinputs*
T0*
_output_shapes
:*
out_type0	f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskG
sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RS
subSubstrided_slice:output:0sub/y:output:0*
T0	*
_output_shapes
: Z
clip_by_value/MinimumMinimumConst:output:0sub:z:0*
T0	*
_output_shapes
: Q
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R n
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*
_output_shapes
: X
GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€§
GatherV2GatherV2inputsclip_by_value:z:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€  @Ґ
max_pooling2d_2/MaxPoolMaxPoolGatherV2:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
I
Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_1Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_1Substrided_slice_1:output:0sub_1/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_1/MinimumMinimumConst_1:output:0	sub_1:z:0*
T0	*
_output_shapes
: S
clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€™

GatherV2_1GatherV2inputsclip_by_value_1:z:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€  @¶
max_pooling2d_2/MaxPool_1MaxPoolGatherV2_1:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_2Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_2StridedSliceShape_2:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_2Substrided_slice_2:output:0sub_2/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_2/MinimumMinimumConst_2:output:0	sub_2:z:0*
T0	*
_output_shapes
: S
clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€™

GatherV2_2GatherV2inputsclip_by_value_2:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€  @¶
max_pooling2d_2/MaxPool_2MaxPoolGatherV2_2:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_3Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€j
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_3StridedSliceShape_3:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_3Substrided_slice_3:output:0sub_3/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_3/MinimumMinimumConst_3:output:0	sub_3:z:0*
T0	*
_output_shapes
: S
clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_3Maximumclip_by_value_3/Minimum:z:0clip_by_value_3/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€™

GatherV2_3GatherV2inputsclip_by_value_3:z:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:€€€€€€€€€  @¶
max_pooling2d_2/MaxPool_3MaxPoolGatherV2_3:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
ч
stackPack max_pooling2d_2/MaxPool:output:0"max_pooling2d_2/MaxPool_1:output:0"max_pooling2d_2/MaxPool_2:output:0"max_pooling2d_2/MaxPool_3:output:0*
N*
T0*3
_output_shapes!
:€€€€€€€€€@*
axisю€€€€€€€€b
IdentityIdentitystack:output:0*
T0*3
_output_shapes!
:€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€  @:[ W
3
_output_shapes!
:€€€€€€€€€  @
 
_user_specified_nameinputs
У
g
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_627959

inputs
identityҐ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs"њL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*“
serving_defaultЊ
c
rot_equiv_conv2d_inputI
(serving_default_rot_equiv_conv2d_input:0€€€€€€€€€РР;
dense_10
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:пн
»
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer-10
layer_with_weights-5
layer-11
layer_with_weights-6
layer-12
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
 
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
	filt_base
bias"
_tf_keras_layer
ѓ
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses
%pool"
_tf_keras_layer
 
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses

,kernel
,	filt_base
-bias"
_tf_keras_layer
ѓ
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses
4pool"
_tf_keras_layer
 
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses

;kernel
;	filt_base
<bias"
_tf_keras_layer
ѓ
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses
Cpool"
_tf_keras_layer
 
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses

Jkernel
J	filt_base
Kbias"
_tf_keras_layer
ѓ
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses
Rpool"
_tf_keras_layer
 
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses

Ykernel
Y	filt_base
Zbias"
_tf_keras_layer
•
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses"
_tf_keras_layer
•
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses"
_tf_keras_layer
ї
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses

mkernel
nbias"
_tf_keras_layer
ї
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses

ukernel
vbias"
_tf_keras_layer
Ж
0
1
,2
-3
;4
<5
J6
K7
Y8
Z9
m10
n11
u12
v13"
trackable_list_wrapper
Ж
0
1
,2
-3
;4
<5
J6
K7
Y8
Z9
m10
n11
u12
v13"
trackable_list_wrapper
 "
trackable_list_wrapper
 
wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
в
|trace_0
}trace_1
~trace_2
trace_32ч
+__inference_sequential_layer_call_fn_625577
+__inference_sequential_layer_call_fn_625991
+__inference_sequential_layer_call_fn_626024
+__inference_sequential_layer_call_fn_625827ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 z|trace_0z}trace_1z~trace_2ztrace_3
÷
Аtrace_0
Бtrace_1
Вtrace_2
Гtrace_32г
F__inference_sequential_layer_call_and_return_conditional_losses_626612
F__inference_sequential_layer_call_and_return_conditional_losses_627200
F__inference_sequential_layer_call_and_return_conditional_losses_625872
F__inference_sequential_layer_call_and_return_conditional_losses_625917ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 zАtrace_0zБtrace_1zВtrace_2zГtrace_3
џBЎ
!__inference__wrapped_model_624800rot_equiv_conv2d_input"Ш
С≤Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Е
	Дiter
Еbeta_1
Жbeta_2

Зdecay
Иlearning_rate
Йmomentum_cachem•m¶,mІ-m®;m©<m™JmЂKmђYm≠ZmЃmmѓnm∞um±vm≤v≥vі,vµ-vґ;vЈ<vЄJvєKvЇYvїZvЉmvљnvЊuvњvvј"
	optimizer
-
Кserving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Лnon_trainable_variables
Мlayers
Нmetrics
 Оlayer_regularization_losses
Пlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ч
Рtrace_02Ў
1__inference_rot_equiv_conv2d_layer_call_fn_627209Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zРtrace_0
Т
Сtrace_02у
L__inference_rot_equiv_conv2d_layer_call_and_return_conditional_losses_627288Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zСtrace_0
1:/ 2rot_equiv_conv2d/kernel
#:! 2rot_equiv_conv2d/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Тnon_trainable_variables
Уlayers
Фmetrics
 Хlayer_regularization_losses
Цlayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
ч
Чtrace_02Ў
1__inference_rot_equiv_pool2d_layer_call_fn_627293Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЧtrace_0
Т
Шtrace_02у
L__inference_rot_equiv_pool2d_layer_call_and_return_conditional_losses_627354Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zШtrace_0
Ђ
Щ	variables
Ъtrainable_variables
Ыregularization_losses
Ь	keras_api
Э__call__
+Ю&call_and_return_all_conditional_losses"
_tf_keras_layer
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Яnon_trainable_variables
†layers
°metrics
 Ґlayer_regularization_losses
£layer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
щ
§trace_02Џ
3__inference_rot_equiv_conv2d_1_layer_call_fn_627363Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z§trace_0
Ф
•trace_02х
N__inference_rot_equiv_conv2d_1_layer_call_and_return_conditional_losses_627433Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z•trace_0
3:1  2rot_equiv_conv2d_1/kernel
%:# 2rot_equiv_conv2d_1/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
¶non_trainable_variables
Іlayers
®metrics
 ©layer_regularization_losses
™layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
щ
Ђtrace_02Џ
3__inference_rot_equiv_pool2d_1_layer_call_fn_627438Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЂtrace_0
Ф
ђtrace_02х
N__inference_rot_equiv_pool2d_1_layer_call_and_return_conditional_losses_627499Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zђtrace_0
Ђ
≠	variables
Ѓtrainable_variables
ѓregularization_losses
∞	keras_api
±__call__
+≤&call_and_return_all_conditional_losses"
_tf_keras_layer
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
≥non_trainable_variables
іlayers
µmetrics
 ґlayer_regularization_losses
Јlayer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
щ
Єtrace_02Џ
3__inference_rot_equiv_conv2d_2_layer_call_fn_627508Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЄtrace_0
Ф
єtrace_02х
N__inference_rot_equiv_conv2d_2_layer_call_and_return_conditional_losses_627578Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zєtrace_0
3:1 @2rot_equiv_conv2d_2/kernel
%:#@2rot_equiv_conv2d_2/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Їnon_trainable_variables
їlayers
Љmetrics
 љlayer_regularization_losses
Њlayer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
щ
њtrace_02Џ
3__inference_rot_equiv_pool2d_2_layer_call_fn_627583Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zњtrace_0
Ф
јtrace_02х
N__inference_rot_equiv_pool2d_2_layer_call_and_return_conditional_losses_627644Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zјtrace_0
Ђ
Ѕ	variables
¬trainable_variables
√regularization_losses
ƒ	keras_api
≈__call__
+∆&call_and_return_all_conditional_losses"
_tf_keras_layer
.
J0
K1"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
«non_trainable_variables
»layers
…metrics
  layer_regularization_losses
Ћlayer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
щ
ћtrace_02Џ
3__inference_rot_equiv_conv2d_3_layer_call_fn_627653Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zћtrace_0
Ф
Ќtrace_02х
N__inference_rot_equiv_conv2d_3_layer_call_and_return_conditional_losses_627723Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЌtrace_0
3:1@@2rot_equiv_conv2d_3/kernel
%:#@2rot_equiv_conv2d_3/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
ќnon_trainable_variables
ѕlayers
–metrics
 —layer_regularization_losses
“layer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
щ
”trace_02Џ
3__inference_rot_equiv_pool2d_3_layer_call_fn_627728Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z”trace_0
Ф
‘trace_02х
N__inference_rot_equiv_pool2d_3_layer_call_and_return_conditional_losses_627789Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z‘trace_0
Ђ
’	variables
÷trainable_variables
„regularization_losses
Ў	keras_api
ў__call__
+Џ&call_and_return_all_conditional_losses"
_tf_keras_layer
.
Y0
Z1"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
џnon_trainable_variables
№layers
Ёmetrics
 ёlayer_regularization_losses
яlayer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
щ
аtrace_02Џ
3__inference_rot_equiv_conv2d_4_layer_call_fn_627798Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zаtrace_0
Ф
бtrace_02х
N__inference_rot_equiv_conv2d_4_layer_call_and_return_conditional_losses_627868Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zбtrace_0
4:2@А2rot_equiv_conv2d_4/kernel
&:$А2rot_equiv_conv2d_4/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
вnon_trainable_variables
гlayers
дmetrics
 еlayer_regularization_losses
жlayer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
у
зtrace_02‘
-__inference_rot_inv_pool_layer_call_fn_627873Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zзtrace_0
О
иtrace_02п
H__inference_rot_inv_pool_layer_call_and_return_conditional_losses_627879Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zиtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
йnon_trainable_variables
кlayers
лmetrics
 мlayer_regularization_losses
нlayer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
о
оtrace_02ѕ
(__inference_flatten_layer_call_fn_627884Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zоtrace_0
Й
пtrace_02к
C__inference_flatten_layer_call_and_return_conditional_losses_627890Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zпtrace_0
.
m0
n1"
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
рnon_trainable_variables
сlayers
тmetrics
 уlayer_regularization_losses
фlayer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
м
хtrace_02Ќ
&__inference_dense_layer_call_fn_627899Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zхtrace_0
З
цtrace_02и
A__inference_dense_layer_call_and_return_conditional_losses_627910Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zцtrace_0
:	А 2dense/kernel
: 2
dense/bias
.
u0
v1"
trackable_list_wrapper
.
u0
v1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
чnon_trainable_variables
шlayers
щmetrics
 ъlayer_regularization_losses
ыlayer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
о
ьtrace_02ѕ
(__inference_dense_1_layer_call_fn_627919Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zьtrace_0
Й
эtrace_02к
C__inference_dense_1_layer_call_and_return_conditional_losses_627929Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zэtrace_0
 : 2dense_1/kernel
:2dense_1/bias
 "
trackable_list_wrapper
~
0
1
2
3
4
5
6
7
	8

9
10
11
12"
trackable_list_wrapper
0
ю0
€1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
НBК
+__inference_sequential_layer_call_fn_625577rot_equiv_conv2d_input"ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
эBъ
+__inference_sequential_layer_call_fn_625991inputs"ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
эBъ
+__inference_sequential_layer_call_fn_626024inputs"ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
НBК
+__inference_sequential_layer_call_fn_625827rot_equiv_conv2d_input"ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ШBХ
F__inference_sequential_layer_call_and_return_conditional_losses_626612inputs"ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ШBХ
F__inference_sequential_layer_call_and_return_conditional_losses_627200inputs"ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
®B•
F__inference_sequential_layer_call_and_return_conditional_losses_625872rot_equiv_conv2d_input"ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
®B•
F__inference_sequential_layer_call_and_return_conditional_losses_625917rot_equiv_conv2d_input"ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
:	 (2
Nadam/iter
: (2Nadam/beta_1
: (2Nadam/beta_2
: (2Nadam/decay
: (2Nadam/learning_rate
: (2Nadam/momentum_cache
ЏB„
$__inference_signature_wrapper_625958rot_equiv_conv2d_input"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
еBв
1__inference_rot_equiv_conv2d_layer_call_fn_627209inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
АBэ
L__inference_rot_equiv_conv2d_layer_call_and_return_conditional_losses_627288inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
'
%0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
еBв
1__inference_rot_equiv_pool2d_layer_call_fn_627293inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
АBэ
L__inference_rot_equiv_pool2d_layer_call_and_return_conditional_losses_627354inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Аnon_trainable_variables
Бlayers
Вmetrics
 Гlayer_regularization_losses
Дlayer_metrics
Щ	variables
Ъtrainable_variables
Ыregularization_losses
Э__call__
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses"
_generic_user_object
ф
Еtrace_02’
.__inference_max_pooling2d_layer_call_fn_627934Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЕtrace_0
П
Жtrace_02р
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_627939Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЖtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
зBд
3__inference_rot_equiv_conv2d_1_layer_call_fn_627363inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ВB€
N__inference_rot_equiv_conv2d_1_layer_call_and_return_conditional_losses_627433inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
'
40"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
зBд
3__inference_rot_equiv_pool2d_1_layer_call_fn_627438inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ВB€
N__inference_rot_equiv_pool2d_1_layer_call_and_return_conditional_losses_627499inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
≠	variables
Ѓtrainable_variables
ѓregularization_losses
±__call__
+≤&call_and_return_all_conditional_losses
'≤"call_and_return_conditional_losses"
_generic_user_object
ц
Мtrace_02„
0__inference_max_pooling2d_1_layer_call_fn_627944Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zМtrace_0
С
Нtrace_02т
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_627949Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zНtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
зBд
3__inference_rot_equiv_conv2d_2_layer_call_fn_627508inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ВB€
N__inference_rot_equiv_conv2d_2_layer_call_and_return_conditional_losses_627578inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
'
C0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
зBд
3__inference_rot_equiv_pool2d_2_layer_call_fn_627583inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ВB€
N__inference_rot_equiv_pool2d_2_layer_call_and_return_conditional_losses_627644inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Оnon_trainable_variables
Пlayers
Рmetrics
 Сlayer_regularization_losses
Тlayer_metrics
Ѕ	variables
¬trainable_variables
√regularization_losses
≈__call__
+∆&call_and_return_all_conditional_losses
'∆"call_and_return_conditional_losses"
_generic_user_object
ц
Уtrace_02„
0__inference_max_pooling2d_2_layer_call_fn_627954Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zУtrace_0
С
Фtrace_02т
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_627959Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zФtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
зBд
3__inference_rot_equiv_conv2d_3_layer_call_fn_627653inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ВB€
N__inference_rot_equiv_conv2d_3_layer_call_and_return_conditional_losses_627723inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
'
R0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
зBд
3__inference_rot_equiv_pool2d_3_layer_call_fn_627728inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ВB€
N__inference_rot_equiv_pool2d_3_layer_call_and_return_conditional_losses_627789inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Хnon_trainable_variables
Цlayers
Чmetrics
 Шlayer_regularization_losses
Щlayer_metrics
’	variables
÷trainable_variables
„regularization_losses
ў__call__
+Џ&call_and_return_all_conditional_losses
'Џ"call_and_return_conditional_losses"
_generic_user_object
ц
Ъtrace_02„
0__inference_max_pooling2d_3_layer_call_fn_627964Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЪtrace_0
С
Ыtrace_02т
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_627969Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЫtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
зBд
3__inference_rot_equiv_conv2d_4_layer_call_fn_627798inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ВB€
N__inference_rot_equiv_conv2d_4_layer_call_and_return_conditional_losses_627868inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
бBё
-__inference_rot_inv_pool_layer_call_fn_627873inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ьBщ
H__inference_rot_inv_pool_layer_call_and_return_conditional_losses_627879inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
№Bў
(__inference_flatten_layer_call_fn_627884inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
чBф
C__inference_flatten_layer_call_and_return_conditional_losses_627890inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЏB„
&__inference_dense_layer_call_fn_627899inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
хBт
A__inference_dense_layer_call_and_return_conditional_losses_627910inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
№Bў
(__inference_dense_1_layer_call_fn_627919inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
чBф
C__inference_dense_1_layer_call_and_return_conditional_losses_627929inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
R
Ь	variables
Э	keras_api

Юtotal

Яcount"
_tf_keras_metric
c
†	variables
°	keras_api

Ґtotal

£count
§
_fn_kwargs"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
вBя
.__inference_max_pooling2d_layer_call_fn_627934inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
эBъ
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_627939inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
дBб
0__inference_max_pooling2d_1_layer_call_fn_627944inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
€Bь
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_627949inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
дBб
0__inference_max_pooling2d_2_layer_call_fn_627954inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
€Bь
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_627959inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
дBб
0__inference_max_pooling2d_3_layer_call_fn_627964inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
€Bь
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_627969inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
0
Ю0
Я1"
trackable_list_wrapper
.
Ь	variables"
_generic_user_object
:  (2total
:  (2count
0
Ґ0
£1"
trackable_list_wrapper
.
†	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
7:5 2Nadam/rot_equiv_conv2d/kernel/m
):' 2Nadam/rot_equiv_conv2d/bias/m
9:7  2!Nadam/rot_equiv_conv2d_1/kernel/m
+:) 2Nadam/rot_equiv_conv2d_1/bias/m
9:7 @2!Nadam/rot_equiv_conv2d_2/kernel/m
+:)@2Nadam/rot_equiv_conv2d_2/bias/m
9:7@@2!Nadam/rot_equiv_conv2d_3/kernel/m
+:)@2Nadam/rot_equiv_conv2d_3/bias/m
::8@А2!Nadam/rot_equiv_conv2d_4/kernel/m
,:*А2Nadam/rot_equiv_conv2d_4/bias/m
%:#	А 2Nadam/dense/kernel/m
: 2Nadam/dense/bias/m
&:$ 2Nadam/dense_1/kernel/m
 :2Nadam/dense_1/bias/m
7:5 2Nadam/rot_equiv_conv2d/kernel/v
):' 2Nadam/rot_equiv_conv2d/bias/v
9:7  2!Nadam/rot_equiv_conv2d_1/kernel/v
+:) 2Nadam/rot_equiv_conv2d_1/bias/v
9:7 @2!Nadam/rot_equiv_conv2d_2/kernel/v
+:)@2Nadam/rot_equiv_conv2d_2/bias/v
9:7@@2!Nadam/rot_equiv_conv2d_3/kernel/v
+:)@2Nadam/rot_equiv_conv2d_3/bias/v
::8@А2!Nadam/rot_equiv_conv2d_4/kernel/v
,:*А2Nadam/rot_equiv_conv2d_4/bias/v
%:#	А 2Nadam/dense/kernel/v
: 2Nadam/dense/bias/v
&:$ 2Nadam/dense_1/kernel/v
 :2Nadam/dense_1/bias/vі
!__inference__wrapped_model_624800О,-;<JKYZmnuvIҐF
?Ґ<
:К7
rot_equiv_conv2d_input€€€€€€€€€РР
™ "1™.
,
dense_1!К
dense_1€€€€€€€€€£
C__inference_dense_1_layer_call_and_return_conditional_losses_627929\uv/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "%Ґ"
К
0€€€€€€€€€
Ъ {
(__inference_dense_1_layer_call_fn_627919Ouv/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "К€€€€€€€€€Ґ
A__inference_dense_layer_call_and_return_conditional_losses_627910]mn0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "%Ґ"
К
0€€€€€€€€€ 
Ъ z
&__inference_dense_layer_call_fn_627899Pmn0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "К€€€€€€€€€ ©
C__inference_flatten_layer_call_and_return_conditional_losses_627890b8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ "&Ґ#
К
0€€€€€€€€€А
Ъ Б
(__inference_flatten_layer_call_fn_627884U8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ "К€€€€€€€€€Ао
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_627949ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ∆
0__inference_max_pooling2d_1_layer_call_fn_627944СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€о
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_627959ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ∆
0__inference_max_pooling2d_2_layer_call_fn_627954СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€о
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_627969ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ∆
0__inference_max_pooling2d_3_layer_call_fn_627964СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€м
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_627939ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ƒ
.__inference_max_pooling2d_layer_call_fn_627934СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€∆
N__inference_rot_equiv_conv2d_1_layer_call_and_return_conditional_losses_627433t,-;Ґ8
1Ґ.
,К)
inputs€€€€€€€€€GG 
™ "1Ґ.
'К$
0€€€€€€€€€EE 
Ъ Ю
3__inference_rot_equiv_conv2d_1_layer_call_fn_627363g,-;Ґ8
1Ґ.
,К)
inputs€€€€€€€€€GG 
™ "$К!€€€€€€€€€EE ∆
N__inference_rot_equiv_conv2d_2_layer_call_and_return_conditional_losses_627578t;<;Ґ8
1Ґ.
,К)
inputs€€€€€€€€€"" 
™ "1Ґ.
'К$
0€€€€€€€€€  @
Ъ Ю
3__inference_rot_equiv_conv2d_2_layer_call_fn_627508g;<;Ґ8
1Ґ.
,К)
inputs€€€€€€€€€"" 
™ "$К!€€€€€€€€€  @∆
N__inference_rot_equiv_conv2d_3_layer_call_and_return_conditional_losses_627723tJK;Ґ8
1Ґ.
,К)
inputs€€€€€€€€€@
™ "1Ґ.
'К$
0€€€€€€€€€@
Ъ Ю
3__inference_rot_equiv_conv2d_3_layer_call_fn_627653gJK;Ґ8
1Ґ.
,К)
inputs€€€€€€€€€@
™ "$К!€€€€€€€€€@«
N__inference_rot_equiv_conv2d_4_layer_call_and_return_conditional_losses_627868uYZ;Ґ8
1Ґ.
,К)
inputs€€€€€€€€€@
™ "2Ґ/
(К%
0€€€€€€€€€А
Ъ Я
3__inference_rot_equiv_conv2d_4_layer_call_fn_627798hYZ;Ґ8
1Ґ.
,К)
inputs€€€€€€€€€@
™ "%К"€€€€€€€€€Аƒ
L__inference_rot_equiv_conv2d_layer_call_and_return_conditional_losses_627288t9Ґ6
/Ґ,
*К'
inputs€€€€€€€€€РР
™ "3Ґ0
)К&
0€€€€€€€€€ОО 
Ъ Ь
1__inference_rot_equiv_conv2d_layer_call_fn_627209g9Ґ6
/Ґ,
*К'
inputs€€€€€€€€€РР
™ "&К#€€€€€€€€€ОО ¬
N__inference_rot_equiv_pool2d_1_layer_call_and_return_conditional_losses_627499p;Ґ8
1Ґ.
,К)
inputs€€€€€€€€€EE 
™ "1Ґ.
'К$
0€€€€€€€€€"" 
Ъ Ъ
3__inference_rot_equiv_pool2d_1_layer_call_fn_627438c;Ґ8
1Ґ.
,К)
inputs€€€€€€€€€EE 
™ "$К!€€€€€€€€€"" ¬
N__inference_rot_equiv_pool2d_2_layer_call_and_return_conditional_losses_627644p;Ґ8
1Ґ.
,К)
inputs€€€€€€€€€  @
™ "1Ґ.
'К$
0€€€€€€€€€@
Ъ Ъ
3__inference_rot_equiv_pool2d_2_layer_call_fn_627583c;Ґ8
1Ґ.
,К)
inputs€€€€€€€€€  @
™ "$К!€€€€€€€€€@¬
N__inference_rot_equiv_pool2d_3_layer_call_and_return_conditional_losses_627789p;Ґ8
1Ґ.
,К)
inputs€€€€€€€€€@
™ "1Ґ.
'К$
0€€€€€€€€€@
Ъ Ъ
3__inference_rot_equiv_pool2d_3_layer_call_fn_627728c;Ґ8
1Ґ.
,К)
inputs€€€€€€€€€@
™ "$К!€€€€€€€€€@¬
L__inference_rot_equiv_pool2d_layer_call_and_return_conditional_losses_627354r=Ґ:
3Ґ0
.К+
inputs€€€€€€€€€ОО 
™ "1Ґ.
'К$
0€€€€€€€€€GG 
Ъ Ъ
1__inference_rot_equiv_pool2d_layer_call_fn_627293e=Ґ:
3Ґ0
.К+
inputs€€€€€€€€€ОО 
™ "$К!€€€€€€€€€GG Ї
H__inference_rot_inv_pool_layer_call_and_return_conditional_losses_627879n<Ґ9
2Ґ/
-К*
inputs€€€€€€€€€А
™ ".Ґ+
$К!
0€€€€€€€€€А
Ъ Т
-__inference_rot_inv_pool_layer_call_fn_627873a<Ґ9
2Ґ/
-К*
inputs€€€€€€€€€А
™ "!К€€€€€€€€€А’
F__inference_sequential_layer_call_and_return_conditional_losses_625872К,-;<JKYZmnuvQҐN
GҐD
:К7
rot_equiv_conv2d_input€€€€€€€€€РР
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ ’
F__inference_sequential_layer_call_and_return_conditional_losses_625917К,-;<JKYZmnuvQҐN
GҐD
:К7
rot_equiv_conv2d_input€€€€€€€€€РР
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ ƒ
F__inference_sequential_layer_call_and_return_conditional_losses_626612z,-;<JKYZmnuvAҐ>
7Ґ4
*К'
inputs€€€€€€€€€РР
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ ƒ
F__inference_sequential_layer_call_and_return_conditional_losses_627200z,-;<JKYZmnuvAҐ>
7Ґ4
*К'
inputs€€€€€€€€€РР
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ ђ
+__inference_sequential_layer_call_fn_625577},-;<JKYZmnuvQҐN
GҐD
:К7
rot_equiv_conv2d_input€€€€€€€€€РР
p 

 
™ "К€€€€€€€€€ђ
+__inference_sequential_layer_call_fn_625827},-;<JKYZmnuvQҐN
GҐD
:К7
rot_equiv_conv2d_input€€€€€€€€€РР
p

 
™ "К€€€€€€€€€Ь
+__inference_sequential_layer_call_fn_625991m,-;<JKYZmnuvAҐ>
7Ґ4
*К'
inputs€€€€€€€€€РР
p 

 
™ "К€€€€€€€€€Ь
+__inference_sequential_layer_call_fn_626024m,-;<JKYZmnuvAҐ>
7Ґ4
*К'
inputs€€€€€€€€€РР
p

 
™ "К€€€€€€€€€—
$__inference_signature_wrapper_625958®,-;<JKYZmnuvcҐ`
Ґ 
Y™V
T
rot_equiv_conv2d_input:К7
rot_equiv_conv2d_input€€€€€€€€€РР"1™.
,
dense_1!К
dense_1€€€€€€€€€