       ŁK"	  ŔsţŘAbrain.Event:2rĚU+z`     ŕ^ĺS	4üsţŘA"íŔ
p
PlaceholderPlaceholder*
dtype0*
shape:˙˙˙˙˙˙˙˙˙*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
R
Placeholder_1Placeholder*
_output_shapes
:*
shape:*
dtype0
R
Placeholder_2Placeholder*
_output_shapes
:*
dtype0*
shape:

*Embedding/Initializer/random_uniform/shapeConst*
valueB"      *
dtype0*
_output_shapes
:*
_class
loc:@Embedding

(Embedding/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
_class
loc:@Embedding*
valueB
 *oGŘ˝

(Embedding/Initializer/random_uniform/maxConst*
dtype0*
_class
loc:@Embedding*
valueB
 *oGŘ=*
_output_shapes
: 
Ý
2Embedding/Initializer/random_uniform/RandomUniformRandomUniform*Embedding/Initializer/random_uniform/shape*
dtype0*
_class
loc:@Embedding*

seed *
_output_shapes
:	*
T0*
seed2 
Â
(Embedding/Initializer/random_uniform/subSub(Embedding/Initializer/random_uniform/max(Embedding/Initializer/random_uniform/min*
T0*
_output_shapes
: *
_class
loc:@Embedding
Ő
(Embedding/Initializer/random_uniform/mulMul2Embedding/Initializer/random_uniform/RandomUniform(Embedding/Initializer/random_uniform/sub*
T0*
_class
loc:@Embedding*
_output_shapes
:	
Ç
$Embedding/Initializer/random_uniformAdd(Embedding/Initializer/random_uniform/mul(Embedding/Initializer/random_uniform/min*
_class
loc:@Embedding*
T0*
_output_shapes
:	

	Embedding
VariableV2*
_output_shapes
:	*
dtype0*
shared_name *
shape:	*
_class
loc:@Embedding*
	container 
ź
Embedding/AssignAssign	Embedding$Embedding/Initializer/random_uniform*
_class
loc:@Embedding*
_output_shapes
:	*
T0*
validate_shape(*
use_locking(
m
Embedding/readIdentity	Embedding*
_class
loc:@Embedding*
T0*
_output_shapes
:	
u
embedding_lookup/axisConst*
value	B : *
_output_shapes
: *
dtype0*
_class
loc:@Embedding
Ű
embedding_lookupGatherV2Embedding/readPlaceholderembedding_lookup/axis*-
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tindices0*
_class
loc:@Embedding*
Taxis0*

batch_dims *
Tparams0
o
embedding_lookup/IdentityIdentityembedding_lookup*-
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
]
DropoutWrapperInit/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
_
DropoutWrapperInit/Const_1Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
_
DropoutWrapperInit/Const_2Const*
dtype0*
_output_shapes
: *
valueB
 *   ?
_
DropoutWrapperInit_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
a
DropoutWrapperInit_1/Const_1Const*
dtype0*
_output_shapes
: *
valueB
 *  ?
a
DropoutWrapperInit_1/Const_2Const*
_output_shapes
: *
valueB
 *   ?*
dtype0

4DropoutWrapperZeroState/BasicLSTMCellZeroState/ConstConst*
dtype0*
_output_shapes
:*
valueB:

6DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_1Const*
dtype0*
valueB:@*
_output_shapes
:
|
:DropoutWrapperZeroState/BasicLSTMCellZeroState/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Ľ
5DropoutWrapperZeroState/BasicLSTMCellZeroState/concatConcatV24DropoutWrapperZeroState/BasicLSTMCellZeroState/Const6DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_1:DropoutWrapperZeroState/BasicLSTMCellZeroState/concat/axis*
_output_shapes
:*

Tidx0*
T0*
N

:DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros/ConstConst*
dtype0*
valueB
 *    *
_output_shapes
: 
ë
4DropoutWrapperZeroState/BasicLSTMCellZeroState/zerosFill5DropoutWrapperZeroState/BasicLSTMCellZeroState/concat:DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros/Const*

index_type0*
T0*
_output_shapes
:	@

6DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_2Const*
dtype0*
valueB:*
_output_shapes
:

6DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_3Const*
_output_shapes
:*
dtype0*
valueB:@

6DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_4Const*
dtype0*
valueB:*
_output_shapes
:

6DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_5Const*
dtype0*
valueB:@*
_output_shapes
:
~
<DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1/axisConst*
value	B : *
_output_shapes
: *
dtype0
Ť
7DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1ConcatV26DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_46DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_5<DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1/axis*

Tidx0*
N*
_output_shapes
:*
T0

<DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
ń
6DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1Fill7DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1<DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1/Const*

index_type0*
T0*
_output_shapes
:	@

6DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_6Const*
dtype0*
_output_shapes
:*
valueB:

6DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_7Const*
valueB:@*
_output_shapes
:*
dtype0

6DropoutWrapperZeroState_1/BasicLSTMCellZeroState/ConstConst*
_output_shapes
:*
dtype0*
valueB:

8DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_1Const*
valueB:@*
dtype0*
_output_shapes
:
~
<DropoutWrapperZeroState_1/BasicLSTMCellZeroState/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
­
7DropoutWrapperZeroState_1/BasicLSTMCellZeroState/concatConcatV26DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const8DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_1<DropoutWrapperZeroState_1/BasicLSTMCellZeroState/concat/axis*
N*
_output_shapes
:*
T0*

Tidx0

<DropoutWrapperZeroState_1/BasicLSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ń
6DropoutWrapperZeroState_1/BasicLSTMCellZeroState/zerosFill7DropoutWrapperZeroState_1/BasicLSTMCellZeroState/concat<DropoutWrapperZeroState_1/BasicLSTMCellZeroState/zeros/Const*

index_type0*
T0*
_output_shapes
:	@

8DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_2Const*
dtype0*
valueB:*
_output_shapes
:

8DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_3Const*
_output_shapes
:*
dtype0*
valueB:@

8DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

8DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_5Const*
_output_shapes
:*
dtype0*
valueB:@

>DropoutWrapperZeroState_1/BasicLSTMCellZeroState/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
ł
9DropoutWrapperZeroState_1/BasicLSTMCellZeroState/concat_1ConcatV28DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_48DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_5>DropoutWrapperZeroState_1/BasicLSTMCellZeroState/concat_1/axis*
N*
T0*
_output_shapes
:*

Tidx0

>DropoutWrapperZeroState_1/BasicLSTMCellZeroState/zeros_1/ConstConst*
dtype0*
valueB
 *    *
_output_shapes
: 
÷
8DropoutWrapperZeroState_1/BasicLSTMCellZeroState/zeros_1Fill9DropoutWrapperZeroState_1/BasicLSTMCellZeroState/concat_1>DropoutWrapperZeroState_1/BasicLSTMCellZeroState/zeros_1/Const*
_output_shapes
:	@*

index_type0*
T0

8DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_6Const*
_output_shapes
:*
dtype0*
valueB:

8DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_7Const*
dtype0*
valueB:@*
_output_shapes
:
^
bidirectional_rnn/fw/fw/RankConst*
dtype0*
value	B :*
_output_shapes
: 
e
#bidirectional_rnn/fw/fw/range/startConst*
dtype0*
_output_shapes
: *
value	B :
e
#bidirectional_rnn/fw/fw/range/deltaConst*
value	B :*
_output_shapes
: *
dtype0
ś
bidirectional_rnn/fw/fw/rangeRange#bidirectional_rnn/fw/fw/range/startbidirectional_rnn/fw/fw/Rank#bidirectional_rnn/fw/fw/range/delta*

Tidx0*
_output_shapes
:
x
'bidirectional_rnn/fw/fw/concat/values_0Const*
valueB"       *
_output_shapes
:*
dtype0
e
#bidirectional_rnn/fw/fw/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ń
bidirectional_rnn/fw/fw/concatConcatV2'bidirectional_rnn/fw/fw/concat/values_0bidirectional_rnn/fw/fw/range#bidirectional_rnn/fw/fw/concat/axis*

Tidx0*
N*
_output_shapes
:*
T0
Ž
!bidirectional_rnn/fw/fw/transpose	Transposeembedding_lookup/Identitybidirectional_rnn/fw/fw/concat*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tperm0
e
'bidirectional_rnn/fw/fw/sequence_lengthIdentityPlaceholder_2*
T0*
_output_shapes
:

bidirectional_rnn/fw/fw/ShapeShape'bidirectional_rnn/fw/fw/sequence_length*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
out_type0*
T0
h
bidirectional_rnn/fw/fw/stackConst*
_output_shapes
:*
valueB:*
dtype0

bidirectional_rnn/fw/fw/EqualEqualbidirectional_rnn/fw/fw/Shapebidirectional_rnn/fw/fw/stack*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
g
bidirectional_rnn/fw/fw/ConstConst*
_output_shapes
:*
dtype0*
valueB: 

bidirectional_rnn/fw/fw/AllAllbidirectional_rnn/fw/fw/Equalbidirectional_rnn/fw/fw/Const*
_output_shapes
: *

Tidx0*
	keep_dims( 
Ź
$bidirectional_rnn/fw/fw/Assert/ConstConst*
_output_shapes
: *X
valueOBM BGExpected shape for Tensor bidirectional_rnn/fw/fw/sequence_length:0 is *
dtype0
w
&bidirectional_rnn/fw/fw/Assert/Const_1Const*
dtype0*
_output_shapes
: *!
valueB B but saw shape: 
´
,bidirectional_rnn/fw/fw/Assert/Assert/data_0Const*
dtype0*
_output_shapes
: *X
valueOBM BGExpected shape for Tensor bidirectional_rnn/fw/fw/sequence_length:0 is 
}
,bidirectional_rnn/fw/fw/Assert/Assert/data_2Const*
dtype0*
_output_shapes
: *!
valueB B but saw shape: 

%bidirectional_rnn/fw/fw/Assert/AssertAssertbidirectional_rnn/fw/fw/All,bidirectional_rnn/fw/fw/Assert/Assert/data_0bidirectional_rnn/fw/fw/stack,bidirectional_rnn/fw/fw/Assert/Assert/data_2bidirectional_rnn/fw/fw/Shape*
T
2*
	summarize
Ł
#bidirectional_rnn/fw/fw/CheckSeqLenIdentity'bidirectional_rnn/fw/fw/sequence_length&^bidirectional_rnn/fw/fw/Assert/Assert*
_output_shapes
:*
T0

bidirectional_rnn/fw/fw/Shape_1Shape!bidirectional_rnn/fw/fw/transpose*
T0*
out_type0*
_output_shapes
:
u
+bidirectional_rnn/fw/fw/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
w
-bidirectional_rnn/fw/fw/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
w
-bidirectional_rnn/fw/fw/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
ó
%bidirectional_rnn/fw/fw/strided_sliceStridedSlicebidirectional_rnn/fw/fw/Shape_1+bidirectional_rnn/fw/fw/strided_slice/stack-bidirectional_rnn/fw/fw/strided_slice/stack_1-bidirectional_rnn/fw/fw/strided_slice/stack_2*
ellipsis_mask *
T0*
end_mask *

begin_mask *
_output_shapes
: *
new_axis_mask *
shrink_axis_mask*
Index0
j
bidirectional_rnn/fw/fw/Const_1Const*
valueB:*
_output_shapes
:*
dtype0
i
bidirectional_rnn/fw/fw/Const_2Const*
dtype0*
valueB:@*
_output_shapes
:
g
%bidirectional_rnn/fw/fw/concat_1/axisConst*
_output_shapes
: *
value	B : *
dtype0
Ď
 bidirectional_rnn/fw/fw/concat_1ConcatV2bidirectional_rnn/fw/fw/Const_1bidirectional_rnn/fw/fw/Const_2%bidirectional_rnn/fw/fw/concat_1/axis*

Tidx0*
T0*
_output_shapes
:*
N
h
#bidirectional_rnn/fw/fw/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
¨
bidirectional_rnn/fw/fw/zerosFill bidirectional_rnn/fw/fw/concat_1#bidirectional_rnn/fw/fw/zeros/Const*
T0*

index_type0*
_output_shapes
:	@
l
bidirectional_rnn/fw/fw/Rank_1Rank#bidirectional_rnn/fw/fw/CheckSeqLen*
_output_shapes
: *
T0
g
%bidirectional_rnn/fw/fw/range_1/startConst*
value	B : *
_output_shapes
: *
dtype0
g
%bidirectional_rnn/fw/fw/range_1/deltaConst*
_output_shapes
: *
value	B :*
dtype0
Ç
bidirectional_rnn/fw/fw/range_1Range%bidirectional_rnn/fw/fw/range_1/startbidirectional_rnn/fw/fw/Rank_1%bidirectional_rnn/fw/fw/range_1/delta*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ś
bidirectional_rnn/fw/fw/MinMin#bidirectional_rnn/fw/fw/CheckSeqLenbidirectional_rnn/fw/fw/range_1*
T0*
	keep_dims( *
_output_shapes
: *

Tidx0
l
bidirectional_rnn/fw/fw/Rank_2Rank#bidirectional_rnn/fw/fw/CheckSeqLen*
_output_shapes
: *
T0
g
%bidirectional_rnn/fw/fw/range_2/startConst*
dtype0*
_output_shapes
: *
value	B : 
g
%bidirectional_rnn/fw/fw/range_2/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
Ç
bidirectional_rnn/fw/fw/range_2Range%bidirectional_rnn/fw/fw/range_2/startbidirectional_rnn/fw/fw/Rank_2%bidirectional_rnn/fw/fw/range_2/delta*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ś
bidirectional_rnn/fw/fw/MaxMax#bidirectional_rnn/fw/fw/CheckSeqLenbidirectional_rnn/fw/fw/range_2*

Tidx0*
_output_shapes
: *
	keep_dims( *
T0
^
bidirectional_rnn/fw/fw/timeConst*
dtype0*
_output_shapes
: *
value	B : 
ľ
#bidirectional_rnn/fw/fw/TensorArrayTensorArrayV3%bidirectional_rnn/fw/fw/strided_slice*
identical_element_shapes(*
element_shape:	@*
dtype0*
_output_shapes

:: *C
tensor_array_name.,bidirectional_rnn/fw/fw/dynamic_rnn/output_0*
clear_after_read(*
dynamic_size( 
ˇ
%bidirectional_rnn/fw/fw/TensorArray_1TensorArrayV3%bidirectional_rnn/fw/fw/strided_slice*
identical_element_shapes(*
dtype0*
element_shape:
*
_output_shapes

:: *
dynamic_size( *B
tensor_array_name-+bidirectional_rnn/fw/fw/dynamic_rnn/input_0*
clear_after_read(

0bidirectional_rnn/fw/fw/TensorArrayUnstack/ShapeShape!bidirectional_rnn/fw/fw/transpose*
_output_shapes
:*
out_type0*
T0

>bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 

@bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:

@bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Đ
8bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_sliceStridedSlice0bidirectional_rnn/fw/fw/TensorArrayUnstack/Shape>bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stack@bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stack_1@bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stack_2*
shrink_axis_mask*
T0*
end_mask *
Index0*

begin_mask *
ellipsis_mask *
_output_shapes
: *
new_axis_mask 
x
6bidirectional_rnn/fw/fw/TensorArrayUnstack/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
x
6bidirectional_rnn/fw/fw/TensorArrayUnstack/range/deltaConst*
_output_shapes
: *
value	B :*
dtype0

0bidirectional_rnn/fw/fw/TensorArrayUnstack/rangeRange6bidirectional_rnn/fw/fw/TensorArrayUnstack/range/start8bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice6bidirectional_rnn/fw/fw/TensorArrayUnstack/range/delta*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ć
Rbidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3%bidirectional_rnn/fw/fw/TensorArray_10bidirectional_rnn/fw/fw/TensorArrayUnstack/range!bidirectional_rnn/fw/fw/transpose'bidirectional_rnn/fw/fw/TensorArray_1:1*
T0*
_output_shapes
: *4
_class*
(&loc:@bidirectional_rnn/fw/fw/transpose
c
!bidirectional_rnn/fw/fw/Maximum/xConst*
_output_shapes
: *
dtype0*
value	B :

bidirectional_rnn/fw/fw/MaximumMaximum!bidirectional_rnn/fw/fw/Maximum/xbidirectional_rnn/fw/fw/Max*
T0*
_output_shapes
: 

bidirectional_rnn/fw/fw/MinimumMinimum%bidirectional_rnn/fw/fw/strided_slicebidirectional_rnn/fw/fw/Maximum*
T0*
_output_shapes
: 
q
/bidirectional_rnn/fw/fw/while/iteration_counterConst*
value	B : *
_output_shapes
: *
dtype0
é
#bidirectional_rnn/fw/fw/while/EnterEnter/bidirectional_rnn/fw/fw/while/iteration_counter*
is_constant( *
T0*
parallel_iterations *;

frame_name-+bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
: 
Ř
%bidirectional_rnn/fw/fw/while/Enter_1Enterbidirectional_rnn/fw/fw/time*
T0*
parallel_iterations *
is_constant( *;

frame_name-+bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
: 
á
%bidirectional_rnn/fw/fw/while/Enter_2Enter%bidirectional_rnn/fw/fw/TensorArray:1*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context*
parallel_iterations *
_output_shapes
: *
T0*
is_constant( 
ů
%bidirectional_rnn/fw/fw/while/Enter_3Enter4DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros*
parallel_iterations *
is_constant( *
T0*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:	@
ű
%bidirectional_rnn/fw/fw/while/Enter_4Enter6DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1*
is_constant( *
parallel_iterations *
T0*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:	@
Ş
#bidirectional_rnn/fw/fw/while/MergeMerge#bidirectional_rnn/fw/fw/while/Enter+bidirectional_rnn/fw/fw/while/NextIteration*
_output_shapes
: : *
T0*
N
°
%bidirectional_rnn/fw/fw/while/Merge_1Merge%bidirectional_rnn/fw/fw/while/Enter_1-bidirectional_rnn/fw/fw/while/NextIteration_1*
_output_shapes
: : *
N*
T0
°
%bidirectional_rnn/fw/fw/while/Merge_2Merge%bidirectional_rnn/fw/fw/while/Enter_2-bidirectional_rnn/fw/fw/while/NextIteration_2*
N*
_output_shapes
: : *
T0
š
%bidirectional_rnn/fw/fw/while/Merge_3Merge%bidirectional_rnn/fw/fw/while/Enter_3-bidirectional_rnn/fw/fw/while/NextIteration_3*
T0*!
_output_shapes
:	@: *
N
š
%bidirectional_rnn/fw/fw/while/Merge_4Merge%bidirectional_rnn/fw/fw/while/Enter_4-bidirectional_rnn/fw/fw/while/NextIteration_4*!
_output_shapes
:	@: *
T0*
N

"bidirectional_rnn/fw/fw/while/LessLess#bidirectional_rnn/fw/fw/while/Merge(bidirectional_rnn/fw/fw/while/Less/Enter*
_output_shapes
: *
T0
ä
(bidirectional_rnn/fw/fw/while/Less/EnterEnter%bidirectional_rnn/fw/fw/strided_slice*
parallel_iterations *
T0*
is_constant(*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
: 
 
$bidirectional_rnn/fw/fw/while/Less_1Less%bidirectional_rnn/fw/fw/while/Merge_1*bidirectional_rnn/fw/fw/while/Less_1/Enter*
_output_shapes
: *
T0
ŕ
*bidirectional_rnn/fw/fw/while/Less_1/EnterEnterbidirectional_rnn/fw/fw/Minimum*
is_constant(*
_output_shapes
: *
T0*
parallel_iterations *;

frame_name-+bidirectional_rnn/fw/fw/while/while_context

(bidirectional_rnn/fw/fw/while/LogicalAnd
LogicalAnd"bidirectional_rnn/fw/fw/while/Less$bidirectional_rnn/fw/fw/while/Less_1*
_output_shapes
: 
t
&bidirectional_rnn/fw/fw/while/LoopCondLoopCond(bidirectional_rnn/fw/fw/while/LogicalAnd*
_output_shapes
: 
Ö
$bidirectional_rnn/fw/fw/while/SwitchSwitch#bidirectional_rnn/fw/fw/while/Merge&bidirectional_rnn/fw/fw/while/LoopCond*6
_class,
*(loc:@bidirectional_rnn/fw/fw/while/Merge*
_output_shapes
: : *
T0
Ü
&bidirectional_rnn/fw/fw/while/Switch_1Switch%bidirectional_rnn/fw/fw/while/Merge_1&bidirectional_rnn/fw/fw/while/LoopCond*
_output_shapes
: : *8
_class.
,*loc:@bidirectional_rnn/fw/fw/while/Merge_1*
T0
Ü
&bidirectional_rnn/fw/fw/while/Switch_2Switch%bidirectional_rnn/fw/fw/while/Merge_2&bidirectional_rnn/fw/fw/while/LoopCond*8
_class.
,*loc:@bidirectional_rnn/fw/fw/while/Merge_2*
_output_shapes
: : *
T0
î
&bidirectional_rnn/fw/fw/while/Switch_3Switch%bidirectional_rnn/fw/fw/while/Merge_3&bidirectional_rnn/fw/fw/while/LoopCond**
_output_shapes
:	@:	@*
T0*8
_class.
,*loc:@bidirectional_rnn/fw/fw/while/Merge_3
î
&bidirectional_rnn/fw/fw/while/Switch_4Switch%bidirectional_rnn/fw/fw/while/Merge_4&bidirectional_rnn/fw/fw/while/LoopCond**
_output_shapes
:	@:	@*8
_class.
,*loc:@bidirectional_rnn/fw/fw/while/Merge_4*
T0
{
&bidirectional_rnn/fw/fw/while/IdentityIdentity&bidirectional_rnn/fw/fw/while/Switch:1*
T0*
_output_shapes
: 

(bidirectional_rnn/fw/fw/while/Identity_1Identity(bidirectional_rnn/fw/fw/while/Switch_1:1*
_output_shapes
: *
T0

(bidirectional_rnn/fw/fw/while/Identity_2Identity(bidirectional_rnn/fw/fw/while/Switch_2:1*
T0*
_output_shapes
: 

(bidirectional_rnn/fw/fw/while/Identity_3Identity(bidirectional_rnn/fw/fw/while/Switch_3:1*
_output_shapes
:	@*
T0

(bidirectional_rnn/fw/fw/while/Identity_4Identity(bidirectional_rnn/fw/fw/while/Switch_4:1*
T0*
_output_shapes
:	@

#bidirectional_rnn/fw/fw/while/add/yConst'^bidirectional_rnn/fw/fw/while/Identity*
_output_shapes
: *
value	B :*
dtype0

!bidirectional_rnn/fw/fw/while/addAdd&bidirectional_rnn/fw/fw/while/Identity#bidirectional_rnn/fw/fw/while/add/y*
_output_shapes
: *
T0

/bidirectional_rnn/fw/fw/while/TensorArrayReadV3TensorArrayReadV35bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter(bidirectional_rnn/fw/fw/while/Identity_17bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1* 
_output_shapes
:
*
dtype0
ő
5bidirectional_rnn/fw/fw/while/TensorArrayReadV3/EnterEnter%bidirectional_rnn/fw/fw/TensorArray_1*
T0*
parallel_iterations *
_output_shapes
:*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context*
is_constant(
 
7bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1EnterRbidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
is_constant(*
_output_shapes
: *;

frame_name-+bidirectional_rnn/fw/fw/while/while_context*
T0*
parallel_iterations 
š
*bidirectional_rnn/fw/fw/while/GreaterEqualGreaterEqual(bidirectional_rnn/fw/fw/while/Identity_10bidirectional_rnn/fw/fw/while/GreaterEqual/Enter*
T0*
_output_shapes
:
ě
0bidirectional_rnn/fw/fw/while/GreaterEqual/EnterEnter#bidirectional_rnn/fw/fw/CheckSeqLen*
T0*
is_constant(*
parallel_iterations *;

frame_name-+bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
Ý
Lbidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/shapeConst*>
_class4
20loc:@bidirectional_rnn/fw/basic_lstm_cell/kernel*
dtype0*
valueB"@     *
_output_shapes
:
Ď
Jbidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *ňę­˝*
dtype0*>
_class4
20loc:@bidirectional_rnn/fw/basic_lstm_cell/kernel
Ď
Jbidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/maxConst*
dtype0*
valueB
 *ňę­=*>
_class4
20loc:@bidirectional_rnn/fw/basic_lstm_cell/kernel*
_output_shapes
: 
Ä
Tbidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformLbidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/shape* 
_output_shapes
:
Ŕ*>
_class4
20loc:@bidirectional_rnn/fw/basic_lstm_cell/kernel*
T0*
dtype0*
seed2 *

seed 
Ę
Jbidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/subSubJbidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/maxJbidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@bidirectional_rnn/fw/basic_lstm_cell/kernel*
_output_shapes
: 
Ţ
Jbidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/mulMulTbidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformJbidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/sub*>
_class4
20loc:@bidirectional_rnn/fw/basic_lstm_cell/kernel* 
_output_shapes
:
Ŕ*
T0
Đ
Fbidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniformAddJbidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/mulJbidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/min*>
_class4
20loc:@bidirectional_rnn/fw/basic_lstm_cell/kernel* 
_output_shapes
:
Ŕ*
T0
ă
+bidirectional_rnn/fw/basic_lstm_cell/kernel
VariableV2* 
_output_shapes
:
Ŕ*
dtype0*
shared_name *>
_class4
20loc:@bidirectional_rnn/fw/basic_lstm_cell/kernel*
shape:
Ŕ*
	container 
Ĺ
2bidirectional_rnn/fw/basic_lstm_cell/kernel/AssignAssign+bidirectional_rnn/fw/basic_lstm_cell/kernelFbidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform*
validate_shape(* 
_output_shapes
:
Ŕ*>
_class4
20loc:@bidirectional_rnn/fw/basic_lstm_cell/kernel*
use_locking(*
T0

0bidirectional_rnn/fw/basic_lstm_cell/kernel/readIdentity+bidirectional_rnn/fw/basic_lstm_cell/kernel* 
_output_shapes
:
Ŕ*
T0
Č
;bidirectional_rnn/fw/basic_lstm_cell/bias/Initializer/zerosConst*
valueB*    *<
_class2
0.loc:@bidirectional_rnn/fw/basic_lstm_cell/bias*
dtype0*
_output_shapes	
:
Ő
)bidirectional_rnn/fw/basic_lstm_cell/bias
VariableV2*
shape:*
dtype0*
shared_name *
_output_shapes	
:*<
_class2
0.loc:@bidirectional_rnn/fw/basic_lstm_cell/bias*
	container 
Ż
0bidirectional_rnn/fw/basic_lstm_cell/bias/AssignAssign)bidirectional_rnn/fw/basic_lstm_cell/bias;bidirectional_rnn/fw/basic_lstm_cell/bias/Initializer/zeros*
use_locking(*<
_class2
0.loc:@bidirectional_rnn/fw/basic_lstm_cell/bias*
T0*
_output_shapes	
:*
validate_shape(

.bidirectional_rnn/fw/basic_lstm_cell/bias/readIdentity)bidirectional_rnn/fw/basic_lstm_cell/bias*
T0*
_output_shapes	
:

3bidirectional_rnn/fw/fw/while/basic_lstm_cell/ConstConst'^bidirectional_rnn/fw/fw/while/Identity*
_output_shapes
: *
value	B :*
dtype0
¤
9bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat/axisConst'^bidirectional_rnn/fw/fw/while/Identity*
_output_shapes
: *
dtype0*
value	B :

4bidirectional_rnn/fw/fw/while/basic_lstm_cell/concatConcatV2/bidirectional_rnn/fw/fw/while/TensorArrayReadV3(bidirectional_rnn/fw/fw/while/Identity_49bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat/axis*
N*

Tidx0*
T0* 
_output_shapes
:
Ŕ

4bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMulMatMul4bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat:bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul/Enter*
T0* 
_output_shapes
:
*
transpose_b( *
transpose_a( 

:bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul/EnterEnter0bidirectional_rnn/fw/basic_lstm_cell/kernel/read*
is_constant(*
parallel_iterations *
T0* 
_output_shapes
:
Ŕ*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context
ő
5bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAddBiasAdd4bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul;bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd/Enter*
data_formatNHWC*
T0* 
_output_shapes
:


;bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd/EnterEnter.bidirectional_rnn/fw/basic_lstm_cell/bias/read*
is_constant(*
T0*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context*
_output_shapes	
:*
parallel_iterations 
 
5bidirectional_rnn/fw/fw/while/basic_lstm_cell/Const_1Const'^bidirectional_rnn/fw/fw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 

3bidirectional_rnn/fw/fw/while/basic_lstm_cell/splitSplit3bidirectional_rnn/fw/fw/while/basic_lstm_cell/Const5bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd*
	num_split*@
_output_shapes.
,:	@:	@:	@:	@*
T0
Ł
5bidirectional_rnn/fw/fw/while/basic_lstm_cell/Const_2Const'^bidirectional_rnn/fw/fw/while/Identity*
valueB
 *    *
dtype0*
_output_shapes
: 
Đ
1bidirectional_rnn/fw/fw/while/basic_lstm_cell/AddAdd5bidirectional_rnn/fw/fw/while/basic_lstm_cell/split:25bidirectional_rnn/fw/fw/while/basic_lstm_cell/Const_2*
_output_shapes
:	@*
T0

5bidirectional_rnn/fw/fw/while/basic_lstm_cell/SigmoidSigmoid1bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add*
T0*
_output_shapes
:	@
Ă
1bidirectional_rnn/fw/fw/while/basic_lstm_cell/MulMul(bidirectional_rnn/fw/fw/while/Identity_35bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid*
T0*
_output_shapes
:	@
Ą
7bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid_1Sigmoid3bidirectional_rnn/fw/fw/while/basic_lstm_cell/split*
_output_shapes
:	@*
T0

2bidirectional_rnn/fw/fw/while/basic_lstm_cell/TanhTanh5bidirectional_rnn/fw/fw/while/basic_lstm_cell/split:1*
_output_shapes
:	@*
T0
Ń
3bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1Mul7bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid_12bidirectional_rnn/fw/fw/while/basic_lstm_cell/Tanh*
_output_shapes
:	@*
T0
Ě
3bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_1Add1bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul3bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1*
T0*
_output_shapes
:	@

4bidirectional_rnn/fw/fw/while/basic_lstm_cell/Tanh_1Tanh3bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_1*
_output_shapes
:	@*
T0
Ł
7bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid_2Sigmoid5bidirectional_rnn/fw/fw/while/basic_lstm_cell/split:3*
_output_shapes
:	@*
T0
Ó
3bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2Mul4bidirectional_rnn/fw/fw/while/basic_lstm_cell/Tanh_17bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid_2*
T0*
_output_shapes
:	@

*bidirectional_rnn/fw/fw/while/dropout/rateConst'^bidirectional_rnn/fw/fw/while/Identity*
_output_shapes
: *
valueB
 *   ?*
dtype0
Ľ
+bidirectional_rnn/fw/fw/while/dropout/ShapeConst'^bidirectional_rnn/fw/fw/while/Identity*
valueB"   @   *
_output_shapes
:*
dtype0
Ś
8bidirectional_rnn/fw/fw/while/dropout/random_uniform/minConst'^bidirectional_rnn/fw/fw/while/Identity*
_output_shapes
: *
dtype0*
valueB
 *    
Ś
8bidirectional_rnn/fw/fw/while/dropout/random_uniform/maxConst'^bidirectional_rnn/fw/fw/while/Identity*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Đ
Bbidirectional_rnn/fw/fw/while/dropout/random_uniform/RandomUniformRandomUniform+bidirectional_rnn/fw/fw/while/dropout/Shape*
T0*
_output_shapes
:	@*
dtype0*
seed2 *

seed 
Ô
8bidirectional_rnn/fw/fw/while/dropout/random_uniform/subSub8bidirectional_rnn/fw/fw/while/dropout/random_uniform/max8bidirectional_rnn/fw/fw/while/dropout/random_uniform/min*
T0*
_output_shapes
: 
ç
8bidirectional_rnn/fw/fw/while/dropout/random_uniform/mulMulBbidirectional_rnn/fw/fw/while/dropout/random_uniform/RandomUniform8bidirectional_rnn/fw/fw/while/dropout/random_uniform/sub*
_output_shapes
:	@*
T0
Ů
4bidirectional_rnn/fw/fw/while/dropout/random_uniformAdd8bidirectional_rnn/fw/fw/while/dropout/random_uniform/mul8bidirectional_rnn/fw/fw/while/dropout/random_uniform/min*
T0*
_output_shapes
:	@

+bidirectional_rnn/fw/fw/while/dropout/sub/xConst'^bidirectional_rnn/fw/fw/while/Identity*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ş
)bidirectional_rnn/fw/fw/while/dropout/subSub+bidirectional_rnn/fw/fw/while/dropout/sub/x*bidirectional_rnn/fw/fw/while/dropout/rate*
T0*
_output_shapes
: 

/bidirectional_rnn/fw/fw/while/dropout/truediv/xConst'^bidirectional_rnn/fw/fw/while/Identity*
dtype0*
valueB
 *  ?*
_output_shapes
: 
ľ
-bidirectional_rnn/fw/fw/while/dropout/truedivRealDiv/bidirectional_rnn/fw/fw/while/dropout/truediv/x)bidirectional_rnn/fw/fw/while/dropout/sub*
_output_shapes
: *
T0
Î
2bidirectional_rnn/fw/fw/while/dropout/GreaterEqualGreaterEqual4bidirectional_rnn/fw/fw/while/dropout/random_uniform*bidirectional_rnn/fw/fw/while/dropout/rate*
_output_shapes
:	@*
T0
ž
)bidirectional_rnn/fw/fw/while/dropout/mulMul3bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2-bidirectional_rnn/fw/fw/while/dropout/truediv*
T0*
_output_shapes
:	@
Ż
*bidirectional_rnn/fw/fw/while/dropout/CastCast2bidirectional_rnn/fw/fw/while/dropout/GreaterEqual*
_output_shapes
:	@*

DstT0*

SrcT0
*
Truncate( 
ł
+bidirectional_rnn/fw/fw/while/dropout/mul_1Mul)bidirectional_rnn/fw/fw/while/dropout/mul*bidirectional_rnn/fw/fw/while/dropout/Cast*
_output_shapes
:	@*
T0

$bidirectional_rnn/fw/fw/while/SelectSelect*bidirectional_rnn/fw/fw/while/GreaterEqual*bidirectional_rnn/fw/fw/while/Select/Enter+bidirectional_rnn/fw/fw/while/dropout/mul_1*>
_class4
20loc:@bidirectional_rnn/fw/fw/while/dropout/mul_1*
_output_shapes
:	@*
T0
§
*bidirectional_rnn/fw/fw/while/Select/EnterEnterbidirectional_rnn/fw/fw/zeros*
parallel_iterations *>
_class4
20loc:@bidirectional_rnn/fw/fw/while/dropout/mul_1*
_output_shapes
:	@*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
T0
­
&bidirectional_rnn/fw/fw/while/Select_1Select*bidirectional_rnn/fw/fw/while/GreaterEqual(bidirectional_rnn/fw/fw/while/Identity_33bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_1*
_output_shapes
:	@*F
_class<
:8loc:@bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_1*
T0
­
&bidirectional_rnn/fw/fw/while/Select_2Select*bidirectional_rnn/fw/fw/while/GreaterEqual(bidirectional_rnn/fw/fw/while/Identity_43bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2*
T0*
_output_shapes
:	@*F
_class<
:8loc:@bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2
ű
Abidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Gbidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3/Enter(bidirectional_rnn/fw/fw/while/Identity_1$bidirectional_rnn/fw/fw/while/Select(bidirectional_rnn/fw/fw/while/Identity_2*
_output_shapes
: *
T0*>
_class4
20loc:@bidirectional_rnn/fw/fw/while/dropout/mul_1
Ĺ
Gbidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnter#bidirectional_rnn/fw/fw/TensorArray*>
_class4
20loc:@bidirectional_rnn/fw/fw/while/dropout/mul_1*
_output_shapes
:*
T0*
is_constant(*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context*
parallel_iterations 

%bidirectional_rnn/fw/fw/while/add_1/yConst'^bidirectional_rnn/fw/fw/while/Identity*
dtype0*
value	B :*
_output_shapes
: 

#bidirectional_rnn/fw/fw/while/add_1Add(bidirectional_rnn/fw/fw/while/Identity_1%bidirectional_rnn/fw/fw/while/add_1/y*
_output_shapes
: *
T0

+bidirectional_rnn/fw/fw/while/NextIterationNextIteration!bidirectional_rnn/fw/fw/while/add*
_output_shapes
: *
T0

-bidirectional_rnn/fw/fw/while/NextIteration_1NextIteration#bidirectional_rnn/fw/fw/while/add_1*
T0*
_output_shapes
: 
˘
-bidirectional_rnn/fw/fw/while/NextIteration_2NextIterationAbidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3*
_output_shapes
: *
T0

-bidirectional_rnn/fw/fw/while/NextIteration_3NextIteration&bidirectional_rnn/fw/fw/while/Select_1*
T0*
_output_shapes
:	@

-bidirectional_rnn/fw/fw/while/NextIteration_4NextIteration&bidirectional_rnn/fw/fw/while/Select_2*
T0*
_output_shapes
:	@
q
"bidirectional_rnn/fw/fw/while/ExitExit$bidirectional_rnn/fw/fw/while/Switch*
_output_shapes
: *
T0
u
$bidirectional_rnn/fw/fw/while/Exit_1Exit&bidirectional_rnn/fw/fw/while/Switch_1*
T0*
_output_shapes
: 
u
$bidirectional_rnn/fw/fw/while/Exit_2Exit&bidirectional_rnn/fw/fw/while/Switch_2*
T0*
_output_shapes
: 
~
$bidirectional_rnn/fw/fw/while/Exit_3Exit&bidirectional_rnn/fw/fw/while/Switch_3*
T0*
_output_shapes
:	@
~
$bidirectional_rnn/fw/fw/while/Exit_4Exit&bidirectional_rnn/fw/fw/while/Switch_4*
_output_shapes
:	@*
T0
ę
:bidirectional_rnn/fw/fw/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3#bidirectional_rnn/fw/fw/TensorArray$bidirectional_rnn/fw/fw/while/Exit_2*
_output_shapes
: *6
_class,
*(loc:@bidirectional_rnn/fw/fw/TensorArray
Ž
4bidirectional_rnn/fw/fw/TensorArrayStack/range/startConst*
dtype0*
_output_shapes
: *6
_class,
*(loc:@bidirectional_rnn/fw/fw/TensorArray*
value	B : 
Ž
4bidirectional_rnn/fw/fw/TensorArrayStack/range/deltaConst*6
_class,
*(loc:@bidirectional_rnn/fw/fw/TensorArray*
dtype0*
value	B :*
_output_shapes
: 
Č
.bidirectional_rnn/fw/fw/TensorArrayStack/rangeRange4bidirectional_rnn/fw/fw/TensorArrayStack/range/start:bidirectional_rnn/fw/fw/TensorArrayStack/TensorArraySizeV34bidirectional_rnn/fw/fw/TensorArrayStack/range/delta*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*6
_class,
*(loc:@bidirectional_rnn/fw/fw/TensorArray*

Tidx0
ß
<bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3#bidirectional_rnn/fw/fw/TensorArray.bidirectional_rnn/fw/fw/TensorArrayStack/range$bidirectional_rnn/fw/fw/while/Exit_2*
dtype0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@*6
_class,
*(loc:@bidirectional_rnn/fw/fw/TensorArray*
element_shape:	@
i
bidirectional_rnn/fw/fw/Const_3Const*
_output_shapes
:*
dtype0*
valueB:@
`
bidirectional_rnn/fw/fw/Rank_3Const*
value	B :*
_output_shapes
: *
dtype0
g
%bidirectional_rnn/fw/fw/range_3/startConst*
_output_shapes
: *
value	B :*
dtype0
g
%bidirectional_rnn/fw/fw/range_3/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
ž
bidirectional_rnn/fw/fw/range_3Range%bidirectional_rnn/fw/fw/range_3/startbidirectional_rnn/fw/fw/Rank_3%bidirectional_rnn/fw/fw/range_3/delta*

Tidx0*
_output_shapes
:
z
)bidirectional_rnn/fw/fw/concat_2/values_0Const*
_output_shapes
:*
dtype0*
valueB"       
g
%bidirectional_rnn/fw/fw/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Ů
 bidirectional_rnn/fw/fw/concat_2ConcatV2)bidirectional_rnn/fw/fw/concat_2/values_0bidirectional_rnn/fw/fw/range_3%bidirectional_rnn/fw/fw/concat_2/axis*
_output_shapes
:*

Tidx0*
N*
T0
Ô
#bidirectional_rnn/fw/fw/transpose_1	Transpose<bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3 bidirectional_rnn/fw/fw/concat_2*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
Tperm0*
T0
Ĺ
$bidirectional_rnn/bw/ReverseSequenceReverseSequenceembedding_lookup/IdentityPlaceholder_2*-
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tlen0*
T0*
	batch_dim *
seq_dim
^
bidirectional_rnn/bw/bw/RankConst*
value	B :*
_output_shapes
: *
dtype0
e
#bidirectional_rnn/bw/bw/range/startConst*
_output_shapes
: *
dtype0*
value	B :
e
#bidirectional_rnn/bw/bw/range/deltaConst*
_output_shapes
: *
value	B :*
dtype0
ś
bidirectional_rnn/bw/bw/rangeRange#bidirectional_rnn/bw/bw/range/startbidirectional_rnn/bw/bw/Rank#bidirectional_rnn/bw/bw/range/delta*

Tidx0*
_output_shapes
:
x
'bidirectional_rnn/bw/bw/concat/values_0Const*
dtype0*
valueB"       *
_output_shapes
:
e
#bidirectional_rnn/bw/bw/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Ń
bidirectional_rnn/bw/bw/concatConcatV2'bidirectional_rnn/bw/bw/concat/values_0bidirectional_rnn/bw/bw/range#bidirectional_rnn/bw/bw/concat/axis*

Tidx0*
N*
T0*
_output_shapes
:
š
!bidirectional_rnn/bw/bw/transpose	Transpose$bidirectional_rnn/bw/ReverseSequencebidirectional_rnn/bw/bw/concat*-
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tperm0*
T0
e
'bidirectional_rnn/bw/bw/sequence_lengthIdentityPlaceholder_2*
_output_shapes
:*
T0

bidirectional_rnn/bw/bw/ShapeShape'bidirectional_rnn/bw/bw/sequence_length*
out_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
h
bidirectional_rnn/bw/bw/stackConst*
valueB:*
dtype0*
_output_shapes
:

bidirectional_rnn/bw/bw/EqualEqualbidirectional_rnn/bw/bw/Shapebidirectional_rnn/bw/bw/stack*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
g
bidirectional_rnn/bw/bw/ConstConst*
dtype0*
valueB: *
_output_shapes
:

bidirectional_rnn/bw/bw/AllAllbidirectional_rnn/bw/bw/Equalbidirectional_rnn/bw/bw/Const*
_output_shapes
: *
	keep_dims( *

Tidx0
Ź
$bidirectional_rnn/bw/bw/Assert/ConstConst*
dtype0*X
valueOBM BGExpected shape for Tensor bidirectional_rnn/bw/bw/sequence_length:0 is *
_output_shapes
: 
w
&bidirectional_rnn/bw/bw/Assert/Const_1Const*
dtype0*
_output_shapes
: *!
valueB B but saw shape: 
´
,bidirectional_rnn/bw/bw/Assert/Assert/data_0Const*
_output_shapes
: *X
valueOBM BGExpected shape for Tensor bidirectional_rnn/bw/bw/sequence_length:0 is *
dtype0
}
,bidirectional_rnn/bw/bw/Assert/Assert/data_2Const*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 

%bidirectional_rnn/bw/bw/Assert/AssertAssertbidirectional_rnn/bw/bw/All,bidirectional_rnn/bw/bw/Assert/Assert/data_0bidirectional_rnn/bw/bw/stack,bidirectional_rnn/bw/bw/Assert/Assert/data_2bidirectional_rnn/bw/bw/Shape*
T
2*
	summarize
Ł
#bidirectional_rnn/bw/bw/CheckSeqLenIdentity'bidirectional_rnn/bw/bw/sequence_length&^bidirectional_rnn/bw/bw/Assert/Assert*
_output_shapes
:*
T0

bidirectional_rnn/bw/bw/Shape_1Shape!bidirectional_rnn/bw/bw/transpose*
out_type0*
T0*
_output_shapes
:
u
+bidirectional_rnn/bw/bw/strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0
w
-bidirectional_rnn/bw/bw/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
w
-bidirectional_rnn/bw/bw/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ó
%bidirectional_rnn/bw/bw/strided_sliceStridedSlicebidirectional_rnn/bw/bw/Shape_1+bidirectional_rnn/bw/bw/strided_slice/stack-bidirectional_rnn/bw/bw/strided_slice/stack_1-bidirectional_rnn/bw/bw/strided_slice/stack_2*
new_axis_mask *
end_mask *
shrink_axis_mask*
Index0*

begin_mask *
_output_shapes
: *
T0*
ellipsis_mask 
j
bidirectional_rnn/bw/bw/Const_1Const*
_output_shapes
:*
dtype0*
valueB:
i
bidirectional_rnn/bw/bw/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@
g
%bidirectional_rnn/bw/bw/concat_1/axisConst*
_output_shapes
: *
value	B : *
dtype0
Ď
 bidirectional_rnn/bw/bw/concat_1ConcatV2bidirectional_rnn/bw/bw/Const_1bidirectional_rnn/bw/bw/Const_2%bidirectional_rnn/bw/bw/concat_1/axis*

Tidx0*
N*
T0*
_output_shapes
:
h
#bidirectional_rnn/bw/bw/zeros/ConstConst*
dtype0*
valueB
 *    *
_output_shapes
: 
¨
bidirectional_rnn/bw/bw/zerosFill bidirectional_rnn/bw/bw/concat_1#bidirectional_rnn/bw/bw/zeros/Const*
_output_shapes
:	@*

index_type0*
T0
l
bidirectional_rnn/bw/bw/Rank_1Rank#bidirectional_rnn/bw/bw/CheckSeqLen*
_output_shapes
: *
T0
g
%bidirectional_rnn/bw/bw/range_1/startConst*
value	B : *
_output_shapes
: *
dtype0
g
%bidirectional_rnn/bw/bw/range_1/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
Ç
bidirectional_rnn/bw/bw/range_1Range%bidirectional_rnn/bw/bw/range_1/startbidirectional_rnn/bw/bw/Rank_1%bidirectional_rnn/bw/bw/range_1/delta*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tidx0
Ś
bidirectional_rnn/bw/bw/MinMin#bidirectional_rnn/bw/bw/CheckSeqLenbidirectional_rnn/bw/bw/range_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
l
bidirectional_rnn/bw/bw/Rank_2Rank#bidirectional_rnn/bw/bw/CheckSeqLen*
_output_shapes
: *
T0
g
%bidirectional_rnn/bw/bw/range_2/startConst*
_output_shapes
: *
value	B : *
dtype0
g
%bidirectional_rnn/bw/bw/range_2/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
Ç
bidirectional_rnn/bw/bw/range_2Range%bidirectional_rnn/bw/bw/range_2/startbidirectional_rnn/bw/bw/Rank_2%bidirectional_rnn/bw/bw/range_2/delta*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tidx0
Ś
bidirectional_rnn/bw/bw/MaxMax#bidirectional_rnn/bw/bw/CheckSeqLenbidirectional_rnn/bw/bw/range_2*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
^
bidirectional_rnn/bw/bw/timeConst*
dtype0*
value	B : *
_output_shapes
: 
ľ
#bidirectional_rnn/bw/bw/TensorArrayTensorArrayV3%bidirectional_rnn/bw/bw/strided_slice*
_output_shapes

:: *
dtype0*
clear_after_read(*
dynamic_size( *
identical_element_shapes(*
element_shape:	@*C
tensor_array_name.,bidirectional_rnn/bw/bw/dynamic_rnn/output_0
ˇ
%bidirectional_rnn/bw/bw/TensorArray_1TensorArrayV3%bidirectional_rnn/bw/bw/strided_slice*
dynamic_size( *
_output_shapes

:: *B
tensor_array_name-+bidirectional_rnn/bw/bw/dynamic_rnn/input_0*
identical_element_shapes(*
dtype0*
element_shape:
*
clear_after_read(

0bidirectional_rnn/bw/bw/TensorArrayUnstack/ShapeShape!bidirectional_rnn/bw/bw/transpose*
_output_shapes
:*
out_type0*
T0

>bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 

@bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0

@bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Đ
8bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_sliceStridedSlice0bidirectional_rnn/bw/bw/TensorArrayUnstack/Shape>bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stack@bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stack_1@bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stack_2*
Index0*
_output_shapes
: *
shrink_axis_mask*
ellipsis_mask *

begin_mask *
end_mask *
T0*
new_axis_mask 
x
6bidirectional_rnn/bw/bw/TensorArrayUnstack/range/startConst*
_output_shapes
: *
value	B : *
dtype0
x
6bidirectional_rnn/bw/bw/TensorArrayUnstack/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :

0bidirectional_rnn/bw/bw/TensorArrayUnstack/rangeRange6bidirectional_rnn/bw/bw/TensorArrayUnstack/range/start8bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice6bidirectional_rnn/bw/bw/TensorArrayUnstack/range/delta*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ć
Rbidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3%bidirectional_rnn/bw/bw/TensorArray_10bidirectional_rnn/bw/bw/TensorArrayUnstack/range!bidirectional_rnn/bw/bw/transpose'bidirectional_rnn/bw/bw/TensorArray_1:1*
_output_shapes
: *4
_class*
(&loc:@bidirectional_rnn/bw/bw/transpose*
T0
c
!bidirectional_rnn/bw/bw/Maximum/xConst*
value	B :*
_output_shapes
: *
dtype0

bidirectional_rnn/bw/bw/MaximumMaximum!bidirectional_rnn/bw/bw/Maximum/xbidirectional_rnn/bw/bw/Max*
_output_shapes
: *
T0

bidirectional_rnn/bw/bw/MinimumMinimum%bidirectional_rnn/bw/bw/strided_slicebidirectional_rnn/bw/bw/Maximum*
T0*
_output_shapes
: 
q
/bidirectional_rnn/bw/bw/while/iteration_counterConst*
dtype0*
_output_shapes
: *
value	B : 
é
#bidirectional_rnn/bw/bw/while/EnterEnter/bidirectional_rnn/bw/bw/while/iteration_counter*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context*
is_constant( *
parallel_iterations *
_output_shapes
: *
T0
Ř
%bidirectional_rnn/bw/bw/while/Enter_1Enterbidirectional_rnn/bw/bw/time*
is_constant( *
T0*
parallel_iterations *;

frame_name-+bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
: 
á
%bidirectional_rnn/bw/bw/while/Enter_2Enter%bidirectional_rnn/bw/bw/TensorArray:1*
parallel_iterations *;

frame_name-+bidirectional_rnn/bw/bw/while/while_context*
T0*
_output_shapes
: *
is_constant( 
ű
%bidirectional_rnn/bw/bw/while/Enter_3Enter6DropoutWrapperZeroState_1/BasicLSTMCellZeroState/zeros*
is_constant( *
parallel_iterations *
_output_shapes
:	@*
T0*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context
ý
%bidirectional_rnn/bw/bw/while/Enter_4Enter8DropoutWrapperZeroState_1/BasicLSTMCellZeroState/zeros_1*
T0*
_output_shapes
:	@*
parallel_iterations *;

frame_name-+bidirectional_rnn/bw/bw/while/while_context*
is_constant( 
Ş
#bidirectional_rnn/bw/bw/while/MergeMerge#bidirectional_rnn/bw/bw/while/Enter+bidirectional_rnn/bw/bw/while/NextIteration*
N*
_output_shapes
: : *
T0
°
%bidirectional_rnn/bw/bw/while/Merge_1Merge%bidirectional_rnn/bw/bw/while/Enter_1-bidirectional_rnn/bw/bw/while/NextIteration_1*
N*
T0*
_output_shapes
: : 
°
%bidirectional_rnn/bw/bw/while/Merge_2Merge%bidirectional_rnn/bw/bw/while/Enter_2-bidirectional_rnn/bw/bw/while/NextIteration_2*
_output_shapes
: : *
T0*
N
š
%bidirectional_rnn/bw/bw/while/Merge_3Merge%bidirectional_rnn/bw/bw/while/Enter_3-bidirectional_rnn/bw/bw/while/NextIteration_3*
T0*!
_output_shapes
:	@: *
N
š
%bidirectional_rnn/bw/bw/while/Merge_4Merge%bidirectional_rnn/bw/bw/while/Enter_4-bidirectional_rnn/bw/bw/while/NextIteration_4*!
_output_shapes
:	@: *
N*
T0

"bidirectional_rnn/bw/bw/while/LessLess#bidirectional_rnn/bw/bw/while/Merge(bidirectional_rnn/bw/bw/while/Less/Enter*
_output_shapes
: *
T0
ä
(bidirectional_rnn/bw/bw/while/Less/EnterEnter%bidirectional_rnn/bw/bw/strided_slice*
parallel_iterations *
T0*
_output_shapes
: *;

frame_name-+bidirectional_rnn/bw/bw/while/while_context*
is_constant(
 
$bidirectional_rnn/bw/bw/while/Less_1Less%bidirectional_rnn/bw/bw/while/Merge_1*bidirectional_rnn/bw/bw/while/Less_1/Enter*
T0*
_output_shapes
: 
ŕ
*bidirectional_rnn/bw/bw/while/Less_1/EnterEnterbidirectional_rnn/bw/bw/Minimum*
is_constant(*
parallel_iterations *;

frame_name-+bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
: *
T0

(bidirectional_rnn/bw/bw/while/LogicalAnd
LogicalAnd"bidirectional_rnn/bw/bw/while/Less$bidirectional_rnn/bw/bw/while/Less_1*
_output_shapes
: 
t
&bidirectional_rnn/bw/bw/while/LoopCondLoopCond(bidirectional_rnn/bw/bw/while/LogicalAnd*
_output_shapes
: 
Ö
$bidirectional_rnn/bw/bw/while/SwitchSwitch#bidirectional_rnn/bw/bw/while/Merge&bidirectional_rnn/bw/bw/while/LoopCond*
T0*
_output_shapes
: : *6
_class,
*(loc:@bidirectional_rnn/bw/bw/while/Merge
Ü
&bidirectional_rnn/bw/bw/while/Switch_1Switch%bidirectional_rnn/bw/bw/while/Merge_1&bidirectional_rnn/bw/bw/while/LoopCond*
T0*
_output_shapes
: : *8
_class.
,*loc:@bidirectional_rnn/bw/bw/while/Merge_1
Ü
&bidirectional_rnn/bw/bw/while/Switch_2Switch%bidirectional_rnn/bw/bw/while/Merge_2&bidirectional_rnn/bw/bw/while/LoopCond*
_output_shapes
: : *
T0*8
_class.
,*loc:@bidirectional_rnn/bw/bw/while/Merge_2
î
&bidirectional_rnn/bw/bw/while/Switch_3Switch%bidirectional_rnn/bw/bw/while/Merge_3&bidirectional_rnn/bw/bw/while/LoopCond*8
_class.
,*loc:@bidirectional_rnn/bw/bw/while/Merge_3**
_output_shapes
:	@:	@*
T0
î
&bidirectional_rnn/bw/bw/while/Switch_4Switch%bidirectional_rnn/bw/bw/while/Merge_4&bidirectional_rnn/bw/bw/while/LoopCond**
_output_shapes
:	@:	@*
T0*8
_class.
,*loc:@bidirectional_rnn/bw/bw/while/Merge_4
{
&bidirectional_rnn/bw/bw/while/IdentityIdentity&bidirectional_rnn/bw/bw/while/Switch:1*
T0*
_output_shapes
: 

(bidirectional_rnn/bw/bw/while/Identity_1Identity(bidirectional_rnn/bw/bw/while/Switch_1:1*
_output_shapes
: *
T0

(bidirectional_rnn/bw/bw/while/Identity_2Identity(bidirectional_rnn/bw/bw/while/Switch_2:1*
T0*
_output_shapes
: 

(bidirectional_rnn/bw/bw/while/Identity_3Identity(bidirectional_rnn/bw/bw/while/Switch_3:1*
_output_shapes
:	@*
T0

(bidirectional_rnn/bw/bw/while/Identity_4Identity(bidirectional_rnn/bw/bw/while/Switch_4:1*
_output_shapes
:	@*
T0

#bidirectional_rnn/bw/bw/while/add/yConst'^bidirectional_rnn/bw/bw/while/Identity*
value	B :*
_output_shapes
: *
dtype0

!bidirectional_rnn/bw/bw/while/addAdd&bidirectional_rnn/bw/bw/while/Identity#bidirectional_rnn/bw/bw/while/add/y*
_output_shapes
: *
T0

/bidirectional_rnn/bw/bw/while/TensorArrayReadV3TensorArrayReadV35bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter(bidirectional_rnn/bw/bw/while/Identity_17bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1*
dtype0* 
_output_shapes
:

ő
5bidirectional_rnn/bw/bw/while/TensorArrayReadV3/EnterEnter%bidirectional_rnn/bw/bw/TensorArray_1*
parallel_iterations *
is_constant(*
T0*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
 
7bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1EnterRbidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
is_constant(*
T0*
_output_shapes
: *
parallel_iterations *;

frame_name-+bidirectional_rnn/bw/bw/while/while_context
š
*bidirectional_rnn/bw/bw/while/GreaterEqualGreaterEqual(bidirectional_rnn/bw/bw/while/Identity_10bidirectional_rnn/bw/bw/while/GreaterEqual/Enter*
_output_shapes
:*
T0
ě
0bidirectional_rnn/bw/bw/while/GreaterEqual/EnterEnter#bidirectional_rnn/bw/bw/CheckSeqLen*
parallel_iterations *
T0*
is_constant(*
_output_shapes
:*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context
Ý
Lbidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/shapeConst*
valueB"@     *>
_class4
20loc:@bidirectional_rnn/bw/basic_lstm_cell/kernel*
_output_shapes
:*
dtype0
Ď
Jbidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/minConst*
valueB
 *ňę­˝*
_output_shapes
: *>
_class4
20loc:@bidirectional_rnn/bw/basic_lstm_cell/kernel*
dtype0
Ď
Jbidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/maxConst*
valueB
 *ňę­=*>
_class4
20loc:@bidirectional_rnn/bw/basic_lstm_cell/kernel*
dtype0*
_output_shapes
: 
Ä
Tbidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformLbidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/shape* 
_output_shapes
:
Ŕ*
seed2 *>
_class4
20loc:@bidirectional_rnn/bw/basic_lstm_cell/kernel*

seed *
T0*
dtype0
Ę
Jbidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/subSubJbidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/maxJbidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *>
_class4
20loc:@bidirectional_rnn/bw/basic_lstm_cell/kernel
Ţ
Jbidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/mulMulTbidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformJbidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
Ŕ*
T0*>
_class4
20loc:@bidirectional_rnn/bw/basic_lstm_cell/kernel
Đ
Fbidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniformAddJbidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/mulJbidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/min*
T0* 
_output_shapes
:
Ŕ*>
_class4
20loc:@bidirectional_rnn/bw/basic_lstm_cell/kernel
ă
+bidirectional_rnn/bw/basic_lstm_cell/kernel
VariableV2*
shared_name * 
_output_shapes
:
Ŕ*
shape:
Ŕ*>
_class4
20loc:@bidirectional_rnn/bw/basic_lstm_cell/kernel*
dtype0*
	container 
Ĺ
2bidirectional_rnn/bw/basic_lstm_cell/kernel/AssignAssign+bidirectional_rnn/bw/basic_lstm_cell/kernelFbidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform* 
_output_shapes
:
Ŕ*
validate_shape(*
use_locking(*>
_class4
20loc:@bidirectional_rnn/bw/basic_lstm_cell/kernel*
T0

0bidirectional_rnn/bw/basic_lstm_cell/kernel/readIdentity+bidirectional_rnn/bw/basic_lstm_cell/kernel*
T0* 
_output_shapes
:
Ŕ
Č
;bidirectional_rnn/bw/basic_lstm_cell/bias/Initializer/zerosConst*<
_class2
0.loc:@bidirectional_rnn/bw/basic_lstm_cell/bias*
_output_shapes	
:*
dtype0*
valueB*    
Ő
)bidirectional_rnn/bw/basic_lstm_cell/bias
VariableV2*
_output_shapes	
:*<
_class2
0.loc:@bidirectional_rnn/bw/basic_lstm_cell/bias*
shared_name *
	container *
shape:*
dtype0
Ż
0bidirectional_rnn/bw/basic_lstm_cell/bias/AssignAssign)bidirectional_rnn/bw/basic_lstm_cell/bias;bidirectional_rnn/bw/basic_lstm_cell/bias/Initializer/zeros*<
_class2
0.loc:@bidirectional_rnn/bw/basic_lstm_cell/bias*
_output_shapes	
:*
validate_shape(*
use_locking(*
T0

.bidirectional_rnn/bw/basic_lstm_cell/bias/readIdentity)bidirectional_rnn/bw/basic_lstm_cell/bias*
_output_shapes	
:*
T0

3bidirectional_rnn/bw/bw/while/basic_lstm_cell/ConstConst'^bidirectional_rnn/bw/bw/while/Identity*
_output_shapes
: *
dtype0*
value	B :
¤
9bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat/axisConst'^bidirectional_rnn/bw/bw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 

4bidirectional_rnn/bw/bw/while/basic_lstm_cell/concatConcatV2/bidirectional_rnn/bw/bw/while/TensorArrayReadV3(bidirectional_rnn/bw/bw/while/Identity_49bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat/axis*

Tidx0* 
_output_shapes
:
Ŕ*
T0*
N

4bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMulMatMul4bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat:bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul/Enter*
transpose_b( *
transpose_a( * 
_output_shapes
:
*
T0

:bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul/EnterEnter0bidirectional_rnn/bw/basic_lstm_cell/kernel/read* 
_output_shapes
:
Ŕ*
parallel_iterations *;

frame_name-+bidirectional_rnn/bw/bw/while/while_context*
T0*
is_constant(
ő
5bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAddBiasAdd4bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul;bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd/Enter* 
_output_shapes
:
*
data_formatNHWC*
T0

;bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd/EnterEnter.bidirectional_rnn/bw/basic_lstm_cell/bias/read*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context*
parallel_iterations *
T0*
_output_shapes	
:*
is_constant(
 
5bidirectional_rnn/bw/bw/while/basic_lstm_cell/Const_1Const'^bidirectional_rnn/bw/bw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 

3bidirectional_rnn/bw/bw/while/basic_lstm_cell/splitSplit3bidirectional_rnn/bw/bw/while/basic_lstm_cell/Const5bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd*
T0*
	num_split*@
_output_shapes.
,:	@:	@:	@:	@
Ł
5bidirectional_rnn/bw/bw/while/basic_lstm_cell/Const_2Const'^bidirectional_rnn/bw/bw/while/Identity*
dtype0*
_output_shapes
: *
valueB
 *    
Đ
1bidirectional_rnn/bw/bw/while/basic_lstm_cell/AddAdd5bidirectional_rnn/bw/bw/while/basic_lstm_cell/split:25bidirectional_rnn/bw/bw/while/basic_lstm_cell/Const_2*
_output_shapes
:	@*
T0

5bidirectional_rnn/bw/bw/while/basic_lstm_cell/SigmoidSigmoid1bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add*
T0*
_output_shapes
:	@
Ă
1bidirectional_rnn/bw/bw/while/basic_lstm_cell/MulMul(bidirectional_rnn/bw/bw/while/Identity_35bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid*
T0*
_output_shapes
:	@
Ą
7bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid_1Sigmoid3bidirectional_rnn/bw/bw/while/basic_lstm_cell/split*
_output_shapes
:	@*
T0

2bidirectional_rnn/bw/bw/while/basic_lstm_cell/TanhTanh5bidirectional_rnn/bw/bw/while/basic_lstm_cell/split:1*
T0*
_output_shapes
:	@
Ń
3bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1Mul7bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid_12bidirectional_rnn/bw/bw/while/basic_lstm_cell/Tanh*
T0*
_output_shapes
:	@
Ě
3bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_1Add1bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul3bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1*
_output_shapes
:	@*
T0

4bidirectional_rnn/bw/bw/while/basic_lstm_cell/Tanh_1Tanh3bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_1*
_output_shapes
:	@*
T0
Ł
7bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid_2Sigmoid5bidirectional_rnn/bw/bw/while/basic_lstm_cell/split:3*
T0*
_output_shapes
:	@
Ó
3bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2Mul4bidirectional_rnn/bw/bw/while/basic_lstm_cell/Tanh_17bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid_2*
_output_shapes
:	@*
T0

*bidirectional_rnn/bw/bw/while/dropout/rateConst'^bidirectional_rnn/bw/bw/while/Identity*
dtype0*
valueB
 *   ?*
_output_shapes
: 
Ľ
+bidirectional_rnn/bw/bw/while/dropout/ShapeConst'^bidirectional_rnn/bw/bw/while/Identity*
dtype0*
valueB"   @   *
_output_shapes
:
Ś
8bidirectional_rnn/bw/bw/while/dropout/random_uniform/minConst'^bidirectional_rnn/bw/bw/while/Identity*
valueB
 *    *
_output_shapes
: *
dtype0
Ś
8bidirectional_rnn/bw/bw/while/dropout/random_uniform/maxConst'^bidirectional_rnn/bw/bw/while/Identity*
_output_shapes
: *
valueB
 *  ?*
dtype0
Đ
Bbidirectional_rnn/bw/bw/while/dropout/random_uniform/RandomUniformRandomUniform+bidirectional_rnn/bw/bw/while/dropout/Shape*

seed *
_output_shapes
:	@*
seed2 *
dtype0*
T0
Ô
8bidirectional_rnn/bw/bw/while/dropout/random_uniform/subSub8bidirectional_rnn/bw/bw/while/dropout/random_uniform/max8bidirectional_rnn/bw/bw/while/dropout/random_uniform/min*
T0*
_output_shapes
: 
ç
8bidirectional_rnn/bw/bw/while/dropout/random_uniform/mulMulBbidirectional_rnn/bw/bw/while/dropout/random_uniform/RandomUniform8bidirectional_rnn/bw/bw/while/dropout/random_uniform/sub*
T0*
_output_shapes
:	@
Ů
4bidirectional_rnn/bw/bw/while/dropout/random_uniformAdd8bidirectional_rnn/bw/bw/while/dropout/random_uniform/mul8bidirectional_rnn/bw/bw/while/dropout/random_uniform/min*
T0*
_output_shapes
:	@

+bidirectional_rnn/bw/bw/while/dropout/sub/xConst'^bidirectional_rnn/bw/bw/while/Identity*
_output_shapes
: *
dtype0*
valueB
 *  ?
Ş
)bidirectional_rnn/bw/bw/while/dropout/subSub+bidirectional_rnn/bw/bw/while/dropout/sub/x*bidirectional_rnn/bw/bw/while/dropout/rate*
T0*
_output_shapes
: 

/bidirectional_rnn/bw/bw/while/dropout/truediv/xConst'^bidirectional_rnn/bw/bw/while/Identity*
dtype0*
_output_shapes
: *
valueB
 *  ?
ľ
-bidirectional_rnn/bw/bw/while/dropout/truedivRealDiv/bidirectional_rnn/bw/bw/while/dropout/truediv/x)bidirectional_rnn/bw/bw/while/dropout/sub*
T0*
_output_shapes
: 
Î
2bidirectional_rnn/bw/bw/while/dropout/GreaterEqualGreaterEqual4bidirectional_rnn/bw/bw/while/dropout/random_uniform*bidirectional_rnn/bw/bw/while/dropout/rate*
_output_shapes
:	@*
T0
ž
)bidirectional_rnn/bw/bw/while/dropout/mulMul3bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2-bidirectional_rnn/bw/bw/while/dropout/truediv*
T0*
_output_shapes
:	@
Ż
*bidirectional_rnn/bw/bw/while/dropout/CastCast2bidirectional_rnn/bw/bw/while/dropout/GreaterEqual*

SrcT0
*
_output_shapes
:	@*

DstT0*
Truncate( 
ł
+bidirectional_rnn/bw/bw/while/dropout/mul_1Mul)bidirectional_rnn/bw/bw/while/dropout/mul*bidirectional_rnn/bw/bw/while/dropout/Cast*
T0*
_output_shapes
:	@

$bidirectional_rnn/bw/bw/while/SelectSelect*bidirectional_rnn/bw/bw/while/GreaterEqual*bidirectional_rnn/bw/bw/while/Select/Enter+bidirectional_rnn/bw/bw/while/dropout/mul_1*>
_class4
20loc:@bidirectional_rnn/bw/bw/while/dropout/mul_1*
_output_shapes
:	@*
T0
§
*bidirectional_rnn/bw/bw/while/Select/EnterEnterbidirectional_rnn/bw/bw/zeros*
is_constant(*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context*
parallel_iterations *
_output_shapes
:	@*
T0*>
_class4
20loc:@bidirectional_rnn/bw/bw/while/dropout/mul_1
­
&bidirectional_rnn/bw/bw/while/Select_1Select*bidirectional_rnn/bw/bw/while/GreaterEqual(bidirectional_rnn/bw/bw/while/Identity_33bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_1*F
_class<
:8loc:@bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_1*
_output_shapes
:	@*
T0
­
&bidirectional_rnn/bw/bw/while/Select_2Select*bidirectional_rnn/bw/bw/while/GreaterEqual(bidirectional_rnn/bw/bw/while/Identity_43bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2*
_output_shapes
:	@*F
_class<
:8loc:@bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2*
T0
ű
Abidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Gbidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3/Enter(bidirectional_rnn/bw/bw/while/Identity_1$bidirectional_rnn/bw/bw/while/Select(bidirectional_rnn/bw/bw/while/Identity_2*
T0*>
_class4
20loc:@bidirectional_rnn/bw/bw/while/dropout/mul_1*
_output_shapes
: 
Ĺ
Gbidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnter#bidirectional_rnn/bw/bw/TensorArray*
parallel_iterations *
_output_shapes
:*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context*
T0*>
_class4
20loc:@bidirectional_rnn/bw/bw/while/dropout/mul_1*
is_constant(

%bidirectional_rnn/bw/bw/while/add_1/yConst'^bidirectional_rnn/bw/bw/while/Identity*
dtype0*
value	B :*
_output_shapes
: 

#bidirectional_rnn/bw/bw/while/add_1Add(bidirectional_rnn/bw/bw/while/Identity_1%bidirectional_rnn/bw/bw/while/add_1/y*
_output_shapes
: *
T0

+bidirectional_rnn/bw/bw/while/NextIterationNextIteration!bidirectional_rnn/bw/bw/while/add*
T0*
_output_shapes
: 

-bidirectional_rnn/bw/bw/while/NextIteration_1NextIteration#bidirectional_rnn/bw/bw/while/add_1*
_output_shapes
: *
T0
˘
-bidirectional_rnn/bw/bw/while/NextIteration_2NextIterationAbidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3*
_output_shapes
: *
T0

-bidirectional_rnn/bw/bw/while/NextIteration_3NextIteration&bidirectional_rnn/bw/bw/while/Select_1*
_output_shapes
:	@*
T0

-bidirectional_rnn/bw/bw/while/NextIteration_4NextIteration&bidirectional_rnn/bw/bw/while/Select_2*
T0*
_output_shapes
:	@
q
"bidirectional_rnn/bw/bw/while/ExitExit$bidirectional_rnn/bw/bw/while/Switch*
_output_shapes
: *
T0
u
$bidirectional_rnn/bw/bw/while/Exit_1Exit&bidirectional_rnn/bw/bw/while/Switch_1*
T0*
_output_shapes
: 
u
$bidirectional_rnn/bw/bw/while/Exit_2Exit&bidirectional_rnn/bw/bw/while/Switch_2*
T0*
_output_shapes
: 
~
$bidirectional_rnn/bw/bw/while/Exit_3Exit&bidirectional_rnn/bw/bw/while/Switch_3*
_output_shapes
:	@*
T0
~
$bidirectional_rnn/bw/bw/while/Exit_4Exit&bidirectional_rnn/bw/bw/while/Switch_4*
_output_shapes
:	@*
T0
ę
:bidirectional_rnn/bw/bw/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3#bidirectional_rnn/bw/bw/TensorArray$bidirectional_rnn/bw/bw/while/Exit_2*6
_class,
*(loc:@bidirectional_rnn/bw/bw/TensorArray*
_output_shapes
: 
Ž
4bidirectional_rnn/bw/bw/TensorArrayStack/range/startConst*6
_class,
*(loc:@bidirectional_rnn/bw/bw/TensorArray*
_output_shapes
: *
value	B : *
dtype0
Ž
4bidirectional_rnn/bw/bw/TensorArrayStack/range/deltaConst*
_output_shapes
: *
dtype0*6
_class,
*(loc:@bidirectional_rnn/bw/bw/TensorArray*
value	B :
Č
.bidirectional_rnn/bw/bw/TensorArrayStack/rangeRange4bidirectional_rnn/bw/bw/TensorArrayStack/range/start:bidirectional_rnn/bw/bw/TensorArrayStack/TensorArraySizeV34bidirectional_rnn/bw/bw/TensorArrayStack/range/delta*

Tidx0*6
_class,
*(loc:@bidirectional_rnn/bw/bw/TensorArray*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ß
<bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3#bidirectional_rnn/bw/bw/TensorArray.bidirectional_rnn/bw/bw/TensorArrayStack/range$bidirectional_rnn/bw/bw/while/Exit_2*
element_shape:	@*
dtype0*6
_class,
*(loc:@bidirectional_rnn/bw/bw/TensorArray*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@
i
bidirectional_rnn/bw/bw/Const_3Const*
_output_shapes
:*
valueB:@*
dtype0
`
bidirectional_rnn/bw/bw/Rank_3Const*
dtype0*
_output_shapes
: *
value	B :
g
%bidirectional_rnn/bw/bw/range_3/startConst*
value	B :*
_output_shapes
: *
dtype0
g
%bidirectional_rnn/bw/bw/range_3/deltaConst*
value	B :*
_output_shapes
: *
dtype0
ž
bidirectional_rnn/bw/bw/range_3Range%bidirectional_rnn/bw/bw/range_3/startbidirectional_rnn/bw/bw/Rank_3%bidirectional_rnn/bw/bw/range_3/delta*
_output_shapes
:*

Tidx0
z
)bidirectional_rnn/bw/bw/concat_2/values_0Const*
dtype0*
_output_shapes
:*
valueB"       
g
%bidirectional_rnn/bw/bw/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Ů
 bidirectional_rnn/bw/bw/concat_2ConcatV2)bidirectional_rnn/bw/bw/concat_2/values_0bidirectional_rnn/bw/bw/range_3%bidirectional_rnn/bw/bw/concat_2/axis*
N*
T0*
_output_shapes
:*

Tidx0
Ô
#bidirectional_rnn/bw/bw/transpose_1	Transpose<bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3 bidirectional_rnn/bw/bw/concat_2*
T0*
Tperm0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@
š
ReverseSequenceReverseSequence#bidirectional_rnn/bw/bw/transpose_1Placeholder_2*

Tlen0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
seq_dim*
	batch_dim *
T0
M
concat/axisConst*
_output_shapes
: *
value	B :*
dtype0
˘
concatConcatV2#bidirectional_rnn/fw/fw/transpose_1ReverseSequenceconcat/axis*-
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*

Tidx0*
N
b
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         
f
ReshapeReshapeconcatReshape/shape*
Tshape0*
T0*$
_output_shapes
:

(Weights/Initializer/random_uniform/shapeConst*
dtype0*
_class
loc:@Weights*
valueB"      *
_output_shapes
:

&Weights/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *ý[ž*
_class
loc:@Weights

&Weights/Initializer/random_uniform/maxConst*
dtype0*
valueB
 *ý[>*
_output_shapes
: *
_class
loc:@Weights
×
0Weights/Initializer/random_uniform/RandomUniformRandomUniform(Weights/Initializer/random_uniform/shape*
dtype0*

seed *
_class
loc:@Weights*
seed2 *
_output_shapes
:	*
T0
ş
&Weights/Initializer/random_uniform/subSub&Weights/Initializer/random_uniform/max&Weights/Initializer/random_uniform/min*
_output_shapes
: *
_class
loc:@Weights*
T0
Í
&Weights/Initializer/random_uniform/mulMul0Weights/Initializer/random_uniform/RandomUniform&Weights/Initializer/random_uniform/sub*
T0*
_output_shapes
:	*
_class
loc:@Weights
ż
"Weights/Initializer/random_uniformAdd&Weights/Initializer/random_uniform/mul&Weights/Initializer/random_uniform/min*
T0*
_output_shapes
:	*
_class
loc:@Weights

Weights
VariableV2*
dtype0*
shared_name *
_output_shapes
:	*
shape:	*
	container *
_class
loc:@Weights
´
Weights/AssignAssignWeights"Weights/Initializer/random_uniform*
_output_shapes
:	*
validate_shape(*
use_locking(*
_class
loc:@Weights*
T0
g
Weights/readIdentityWeights*
T0*
_output_shapes
:	*
_class
loc:@Weights

%Bias/Initializer/random_uniform/shapeConst*
_output_shapes
:*
dtype0*
_class
	loc:@Bias*
valueB:

#Bias/Initializer/random_uniform/minConst*
dtype0*
_class
	loc:@Bias*
valueB
 *qÄż*
_output_shapes
: 

#Bias/Initializer/random_uniform/maxConst*
_class
	loc:@Bias*
valueB
 *qÄ?*
_output_shapes
: *
dtype0
É
-Bias/Initializer/random_uniform/RandomUniformRandomUniform%Bias/Initializer/random_uniform/shape*
T0*
dtype0*
_output_shapes
:*
seed2 *

seed *
_class
	loc:@Bias
Ž
#Bias/Initializer/random_uniform/subSub#Bias/Initializer/random_uniform/max#Bias/Initializer/random_uniform/min*
T0*
_output_shapes
: *
_class
	loc:@Bias
ź
#Bias/Initializer/random_uniform/mulMul-Bias/Initializer/random_uniform/RandomUniform#Bias/Initializer/random_uniform/sub*
T0*
_class
	loc:@Bias*
_output_shapes
:
Ž
Bias/Initializer/random_uniformAdd#Bias/Initializer/random_uniform/mul#Bias/Initializer/random_uniform/min*
_class
	loc:@Bias*
T0*
_output_shapes
:

Bias
VariableV2*
shape:*
dtype0*
_class
	loc:@Bias*
shared_name *
	container *
_output_shapes
:
Ł
Bias/AssignAssignBiasBias/Initializer/random_uniform*
validate_shape(*
_class
	loc:@Bias*
use_locking(*
T0*
_output_shapes
:
Y
	Bias/readIdentityBias*
T0*
_class
	loc:@Bias*
_output_shapes
:
v
MatMulBatchMatMulV2ReshapeWeights/read*
adj_x( *#
_output_shapes
:*
adj_y( *
T0
K
addAddMatMul	Bias/read*#
_output_shapes
:*
T0
R
ArgMax/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
p
ArgMaxArgMaxaddArgMax/dimension*
output_type0	*
_output_shapes
:	*

Tidx0*
T0
R
one_hot/on_valueConst*
dtype0*
_output_shapes
: *
value	B :
S
one_hot/off_valueConst*
dtype0*
_output_shapes
: *
value	B : 
O
one_hot/depthConst*
dtype0*
_output_shapes
: *
value	B :

one_hotOneHotPlaceholder_1one_hot/depthone_hot/on_valueone_hot/off_value*
T0*
_output_shapes
:*
axis˙˙˙˙˙˙˙˙˙*
TI0
d
Reshape_1/shapeConst*!
valueB"         *
_output_shapes
:*
dtype0
j
	Reshape_1Reshapeone_hotReshape_1/shape*
T0*
Tshape0*#
_output_shapes
:

9softmax_cross_entropy_with_logits_sg/labels_stop_gradientStopGradient	Reshape_1*
T0*#
_output_shapes
:
š
)softmax_cross_entropy_with_logits_sg/CastCast9softmax_cross_entropy_with_logits_sg/labels_stop_gradient*#
_output_shapes
:*

SrcT0*
Truncate( *

DstT0
k
)softmax_cross_entropy_with_logits_sg/RankConst*
dtype0*
value	B :*
_output_shapes
: 

*softmax_cross_entropy_with_logits_sg/ShapeConst*
_output_shapes
:*!
valueB"         *
dtype0
m
+softmax_cross_entropy_with_logits_sg/Rank_1Const*
_output_shapes
: *
value	B :*
dtype0

,softmax_cross_entropy_with_logits_sg/Shape_1Const*!
valueB"         *
_output_shapes
:*
dtype0
l
*softmax_cross_entropy_with_logits_sg/Sub/yConst*
_output_shapes
: *
dtype0*
value	B :
Š
(softmax_cross_entropy_with_logits_sg/SubSub+softmax_cross_entropy_with_logits_sg/Rank_1*softmax_cross_entropy_with_logits_sg/Sub/y*
_output_shapes
: *
T0

0softmax_cross_entropy_with_logits_sg/Slice/beginPack(softmax_cross_entropy_with_logits_sg/Sub*
_output_shapes
:*

axis *
N*
T0
y
/softmax_cross_entropy_with_logits_sg/Slice/sizeConst*
valueB:*
_output_shapes
:*
dtype0
ö
*softmax_cross_entropy_with_logits_sg/SliceSlice,softmax_cross_entropy_with_logits_sg/Shape_10softmax_cross_entropy_with_logits_sg/Slice/begin/softmax_cross_entropy_with_logits_sg/Slice/size*
Index0*
_output_shapes
:*
T0

4softmax_cross_entropy_with_logits_sg/concat/values_0Const*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
r
0softmax_cross_entropy_with_logits_sg/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 

+softmax_cross_entropy_with_logits_sg/concatConcatV24softmax_cross_entropy_with_logits_sg/concat/values_0*softmax_cross_entropy_with_logits_sg/Slice0softmax_cross_entropy_with_logits_sg/concat/axis*

Tidx0*
_output_shapes
:*
N*
T0
˘
,softmax_cross_entropy_with_logits_sg/ReshapeReshapeadd+softmax_cross_entropy_with_logits_sg/concat*
T0* 
_output_shapes
:
 *
Tshape0
m
+softmax_cross_entropy_with_logits_sg/Rank_2Const*
dtype0*
_output_shapes
: *
value	B :

,softmax_cross_entropy_with_logits_sg/Shape_2Const*!
valueB"         *
dtype0*
_output_shapes
:
n
,softmax_cross_entropy_with_logits_sg/Sub_1/yConst*
value	B :*
_output_shapes
: *
dtype0
­
*softmax_cross_entropy_with_logits_sg/Sub_1Sub+softmax_cross_entropy_with_logits_sg/Rank_2,softmax_cross_entropy_with_logits_sg/Sub_1/y*
T0*
_output_shapes
: 
 
2softmax_cross_entropy_with_logits_sg/Slice_1/beginPack*softmax_cross_entropy_with_logits_sg/Sub_1*

axis *
N*
T0*
_output_shapes
:
{
1softmax_cross_entropy_with_logits_sg/Slice_1/sizeConst*
dtype0*
_output_shapes
:*
valueB:
ü
,softmax_cross_entropy_with_logits_sg/Slice_1Slice,softmax_cross_entropy_with_logits_sg/Shape_22softmax_cross_entropy_with_logits_sg/Slice_1/begin1softmax_cross_entropy_with_logits_sg/Slice_1/size*
Index0*
T0*
_output_shapes
:

6softmax_cross_entropy_with_logits_sg/concat_1/values_0Const*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
t
2softmax_cross_entropy_with_logits_sg/concat_1/axisConst*
dtype0*
value	B : *
_output_shapes
: 

-softmax_cross_entropy_with_logits_sg/concat_1ConcatV26softmax_cross_entropy_with_logits_sg/concat_1/values_0,softmax_cross_entropy_with_logits_sg/Slice_12softmax_cross_entropy_with_logits_sg/concat_1/axis*
_output_shapes
:*
N*
T0*

Tidx0
Ě
.softmax_cross_entropy_with_logits_sg/Reshape_1Reshape)softmax_cross_entropy_with_logits_sg/Cast-softmax_cross_entropy_with_logits_sg/concat_1*
Tshape0* 
_output_shapes
:
 *
T0
Ö
$softmax_cross_entropy_with_logits_sgSoftmaxCrossEntropyWithLogits,softmax_cross_entropy_with_logits_sg/Reshape.softmax_cross_entropy_with_logits_sg/Reshape_1*(
_output_shapes
: :
 *
T0
n
,softmax_cross_entropy_with_logits_sg/Sub_2/yConst*
dtype0*
value	B :*
_output_shapes
: 
Ť
*softmax_cross_entropy_with_logits_sg/Sub_2Sub)softmax_cross_entropy_with_logits_sg/Rank,softmax_cross_entropy_with_logits_sg/Sub_2/y*
_output_shapes
: *
T0
|
2softmax_cross_entropy_with_logits_sg/Slice_2/beginConst*
dtype0*
valueB: *
_output_shapes
:

1softmax_cross_entropy_with_logits_sg/Slice_2/sizePack*softmax_cross_entropy_with_logits_sg/Sub_2*
T0*
_output_shapes
:*
N*

axis 
ú
,softmax_cross_entropy_with_logits_sg/Slice_2Slice*softmax_cross_entropy_with_logits_sg/Shape2softmax_cross_entropy_with_logits_sg/Slice_2/begin1softmax_cross_entropy_with_logits_sg/Slice_2/size*
_output_shapes
:*
Index0*
T0
Ĺ
.softmax_cross_entropy_with_logits_sg/Reshape_2Reshape$softmax_cross_entropy_with_logits_sg,softmax_cross_entropy_with_logits_sg/Slice_2*
T0*
_output_shapes
:	*
Tshape0
`
gradients/ShapeConst*
valueB"      *
_output_shapes
:*
dtype0
X
gradients/grad_ys_0Const*
valueB
 *  ?*
_output_shapes
: *
dtype0
x
gradients/FillFillgradients/Shapegradients/grad_ys_0*

index_type0*
_output_shapes
:	*
T0
S
gradients/f_countConst*
_output_shapes
: *
value	B : *
dtype0
ť
gradients/f_count_1Entergradients/f_count*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context*
is_constant( *
parallel_iterations *
_output_shapes
: *
T0
r
gradients/MergeMergegradients/f_count_1gradients/NextIteration*
N*
T0*
_output_shapes
: : 
v
gradients/SwitchSwitchgradients/Merge&bidirectional_rnn/fw/fw/while/LoopCond*
_output_shapes
: : *
T0
z
gradients/Add/yConst'^bidirectional_rnn/fw/fw/while/Identity*
_output_shapes
: *
dtype0*
value	B :
Z
gradients/AddAddgradients/Switch:1gradients/Add/y*
_output_shapes
: *
T0
Ł	
gradients/NextIterationNextIterationgradients/AddI^gradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPushV2o^gradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2Y^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2S^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul/StackPushV2U^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPushV2S^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul/StackPushV2U^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPushV2Q^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul/StackPushV2S^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul_1/StackPushV2K^gradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul/StackPushV2M^gradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul_1/StackPushV2I^gradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/StackPushV2K^gradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/StackPushV2*
_output_shapes
: *
T0
N
gradients/f_count_2Exitgradients/Switch*
_output_shapes
: *
T0
S
gradients/b_countConst*
value	B :*
_output_shapes
: *
dtype0
Ç
gradients/b_count_1Entergradients/f_count_2*
parallel_iterations *
is_constant( *E

frame_name75gradients/bidirectional_rnn/fw/fw/while/while_context*
T0*
_output_shapes
: 
v
gradients/Merge_1Mergegradients/b_count_1gradients/NextIteration_1*
T0*
_output_shapes
: : *
N
x
gradients/GreaterEqualGreaterEqualgradients/Merge_1gradients/GreaterEqual/Enter*
_output_shapes
: *
T0
Î
gradients/GreaterEqual/EnterEntergradients/b_count*
is_constant(*
T0*E

frame_name75gradients/bidirectional_rnn/fw/fw/while/while_context*
parallel_iterations *
_output_shapes
: 
O
gradients/b_count_2LoopCondgradients/GreaterEqual*
_output_shapes
: 
g
gradients/Switch_1Switchgradients/Merge_1gradients/b_count_2*
_output_shapes
: : *
T0
i
gradients/SubSubgradients/Switch_1:1gradients/GreaterEqual/Enter*
_output_shapes
: *
T0
Ć
gradients/NextIteration_1NextIterationgradients/Subj^gradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_sync*
_output_shapes
: *
T0
P
gradients/b_count_3Exitgradients/Switch_1*
T0*
_output_shapes
: 
U
gradients/f_count_3Const*
value	B : *
_output_shapes
: *
dtype0
˝
gradients/f_count_4Entergradients/f_count_3*
T0*
is_constant( *;

frame_name-+bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
: *
parallel_iterations 
v
gradients/Merge_2Mergegradients/f_count_4gradients/NextIteration_2*
N*
T0*
_output_shapes
: : 
z
gradients/Switch_2Switchgradients/Merge_2&bidirectional_rnn/bw/bw/while/LoopCond*
T0*
_output_shapes
: : 
|
gradients/Add_1/yConst'^bidirectional_rnn/bw/bw/while/Identity*
_output_shapes
: *
value	B :*
dtype0
`
gradients/Add_1Addgradients/Switch_2:1gradients/Add_1/y*
_output_shapes
: *
T0
§	
gradients/NextIteration_2NextIterationgradients/Add_1I^gradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPushV2o^gradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2Y^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2S^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul/StackPushV2U^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPushV2S^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul/StackPushV2U^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPushV2Q^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul/StackPushV2S^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul_1/StackPushV2K^gradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul/StackPushV2M^gradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul_1/StackPushV2I^gradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/StackPushV2K^gradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/StackPushV2*
T0*
_output_shapes
: 
P
gradients/f_count_5Exitgradients/Switch_2*
_output_shapes
: *
T0
U
gradients/b_count_4Const*
value	B :*
dtype0*
_output_shapes
: 
Ç
gradients/b_count_5Entergradients/f_count_5*
parallel_iterations *
_output_shapes
: *E

frame_name75gradients/bidirectional_rnn/bw/bw/while/while_context*
T0*
is_constant( 
v
gradients/Merge_3Mergegradients/b_count_5gradients/NextIteration_3*
T0*
N*
_output_shapes
: : 
|
gradients/GreaterEqual_1GreaterEqualgradients/Merge_3gradients/GreaterEqual_1/Enter*
T0*
_output_shapes
: 
Ň
gradients/GreaterEqual_1/EnterEntergradients/b_count_4*E

frame_name75gradients/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
: *
T0
Q
gradients/b_count_6LoopCondgradients/GreaterEqual_1*
_output_shapes
: 
g
gradients/Switch_3Switchgradients/Merge_3gradients/b_count_6*
T0*
_output_shapes
: : 
m
gradients/Sub_1Subgradients/Switch_3:1gradients/GreaterEqual_1/Enter*
_output_shapes
: *
T0
Č
gradients/NextIteration_3NextIterationgradients/Sub_1j^gradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_sync*
_output_shapes
: *
T0
P
gradients/b_count_7Exitgradients/Switch_3*
T0*
_output_shapes
: 

Cgradients/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 
Ú
Egradients/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeReshapegradients/FillCgradients/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/Shape*
Tshape0*
T0*
_output_shapes

: 
t
gradients/zeros_like	ZerosLike&softmax_cross_entropy_with_logits_sg:1* 
_output_shapes
:
 *
T0

Bgradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙

>gradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims
ExpandDimsEgradients/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeBgradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dim*

Tdim0*
T0* 
_output_shapes
:
 
Ń
7gradients/softmax_cross_entropy_with_logits_sg_grad/mulMul>gradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims&softmax_cross_entropy_with_logits_sg:1* 
_output_shapes
:
 *
T0
Ľ
>gradients/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax
LogSoftmax,softmax_cross_entropy_with_logits_sg/Reshape* 
_output_shapes
:
 *
T0
Š
7gradients/softmax_cross_entropy_with_logits_sg_grad/NegNeg>gradients/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax*
T0* 
_output_shapes
:
 

Dgradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙

@gradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1
ExpandDimsEgradients/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeDgradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dim*
T0* 
_output_shapes
:
 *

Tdim0
ć
9gradients/softmax_cross_entropy_with_logits_sg_grad/mul_1Mul@gradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_17gradients/softmax_cross_entropy_with_logits_sg_grad/Neg*
T0* 
_output_shapes
:
 
Â
Dgradients/softmax_cross_entropy_with_logits_sg_grad/tuple/group_depsNoOp8^gradients/softmax_cross_entropy_with_logits_sg_grad/mul:^gradients/softmax_cross_entropy_with_logits_sg_grad/mul_1
Ď
Lgradients/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependencyIdentity7gradients/softmax_cross_entropy_with_logits_sg_grad/mulE^gradients/softmax_cross_entropy_with_logits_sg_grad/tuple/group_deps* 
_output_shapes
:
 *J
_class@
><loc:@gradients/softmax_cross_entropy_with_logits_sg_grad/mul*
T0
Ő
Ngradients/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependency_1Identity9gradients/softmax_cross_entropy_with_logits_sg_grad/mul_1E^gradients/softmax_cross_entropy_with_logits_sg_grad/tuple/group_deps* 
_output_shapes
:
 *
T0*L
_classB
@>loc:@gradients/softmax_cross_entropy_with_logits_sg_grad/mul_1

Agradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/ShapeConst*!
valueB"         *
dtype0*
_output_shapes
:

Cgradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/ReshapeReshapeLgradients/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependencyAgradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/Shape*#
_output_shapes
:*
Tshape0*
T0
m
gradients/add_grad/ShapeConst*
dtype0*!
valueB"         *
_output_shapes
:
d
gradients/add_grad/Shape_1Const*
_output_shapes
:*
valueB:*
dtype0
´
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
×
gradients/add_grad/SumSumCgradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape(gradients/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*#
_output_shapes
:

gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
Tshape0*#
_output_shapes
:*
T0
Ň
gradients/add_grad/Sum_1SumCgradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape*gradients/add_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0

gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
Ö
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*-
_class#
!loc:@gradients/add_grad/Reshape*
T0*#
_output_shapes
:
Ó
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
_output_shapes
:
ą
gradients/MatMul_grad/MatMulBatchMatMulV2+gradients/add_grad/tuple/control_dependencyWeights/read*$
_output_shapes
:*
adj_x( *
T0*
adj_y(
Ž
gradients/MatMul_grad/MatMul_1BatchMatMulV2Reshape+gradients/add_grad/tuple/control_dependency*
T0*$
_output_shapes
:*
adj_x(*
adj_y( 
p
gradients/MatMul_grad/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"         
n
gradients/MatMul_grad/Shape_1Const*
valueB"      *
_output_shapes
:*
dtype0
s
)gradients/MatMul_grad/strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0
~
+gradients/MatMul_grad/strided_slice/stack_1Const*
dtype0*
valueB:
ţ˙˙˙˙˙˙˙˙*
_output_shapes
:
u
+gradients/MatMul_grad/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ë
#gradients/MatMul_grad/strided_sliceStridedSlicegradients/MatMul_grad/Shape)gradients/MatMul_grad/strided_slice/stack+gradients/MatMul_grad/strided_slice/stack_1+gradients/MatMul_grad/strided_slice/stack_2*
new_axis_mask *
ellipsis_mask *
Index0*

begin_mask*
end_mask *
T0*
_output_shapes
:*
shrink_axis_mask 
u
+gradients/MatMul_grad/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:

-gradients/MatMul_grad/strided_slice_1/stack_1Const*
dtype0*
valueB:
ţ˙˙˙˙˙˙˙˙*
_output_shapes
:
w
-gradients/MatMul_grad/strided_slice_1/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
ó
%gradients/MatMul_grad/strided_slice_1StridedSlicegradients/MatMul_grad/Shape_1+gradients/MatMul_grad/strided_slice_1/stack-gradients/MatMul_grad/strided_slice_1/stack_1-gradients/MatMul_grad/strided_slice_1/stack_2*
Index0*
ellipsis_mask *
new_axis_mask *
T0*
_output_shapes
: *
shrink_axis_mask *

begin_mask*
end_mask 
Í
+gradients/MatMul_grad/BroadcastGradientArgsBroadcastGradientArgs#gradients/MatMul_grad/strided_slice%gradients/MatMul_grad/strided_slice_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ˇ
gradients/MatMul_grad/SumSumgradients/MatMul_grad/MatMul+gradients/MatMul_grad/BroadcastGradientArgs*
T0*
	keep_dims( *$
_output_shapes
:*

Tidx0

gradients/MatMul_grad/ReshapeReshapegradients/MatMul_grad/Sumgradients/MatMul_grad/Shape*
Tshape0*
T0*$
_output_shapes
:
¸
gradients/MatMul_grad/Sum_1Sumgradients/MatMul_grad/MatMul_1-gradients/MatMul_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:	

gradients/MatMul_grad/Reshape_1Reshapegradients/MatMul_grad/Sum_1gradients/MatMul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:	
p
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/Reshape ^gradients/MatMul_grad/Reshape_1
ă
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/Reshape'^gradients/MatMul_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/MatMul_grad/Reshape*$
_output_shapes
:
ä
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/Reshape_1'^gradients/MatMul_grad/tuple/group_deps*2
_class(
&$loc:@gradients/MatMul_grad/Reshape_1*
_output_shapes
:	*
T0
b
gradients/Reshape_grad/ShapeShapeconcat*
T0*
out_type0*
_output_shapes
:
´
gradients/Reshape_grad/ReshapeReshape.gradients/MatMul_grad/tuple/control_dependencygradients/Reshape_grad/Shape*
Tshape0*
T0*$
_output_shapes
:
\
gradients/concat_grad/RankConst*
dtype0*
value	B :*
_output_shapes
: 
o
gradients/concat_grad/modFloorModconcat/axisgradients/concat_grad/Rank*
_output_shapes
: *
T0
~
gradients/concat_grad/ShapeShape#bidirectional_rnn/fw/fw/transpose_1*
out_type0*
_output_shapes
:*
T0
 
gradients/concat_grad/ShapeNShapeN#bidirectional_rnn/fw/fw/transpose_1ReverseSequence*
T0*
out_type0*
N* 
_output_shapes
::
ś
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/modgradients/concat_grad/ShapeNgradients/concat_grad/ShapeN:1*
N* 
_output_shapes
::
Ę
gradients/concat_grad/SliceSlicegradients/Reshape_grad/Reshape"gradients/concat_grad/ConcatOffsetgradients/concat_grad/ShapeN*
T0*
Index0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@
Đ
gradients/concat_grad/Slice_1Slicegradients/Reshape_grad/Reshape$gradients/concat_grad/ConcatOffset:1gradients/concat_grad/ShapeN:1*
Index0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
T0
l
&gradients/concat_grad/tuple/group_depsNoOp^gradients/concat_grad/Slice^gradients/concat_grad/Slice_1
ç
.gradients/concat_grad/tuple/control_dependencyIdentitygradients/concat_grad/Slice'^gradients/concat_grad/tuple/group_deps*.
_class$
" loc:@gradients/concat_grad/Slice*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
T0
í
0gradients/concat_grad/tuple/control_dependency_1Identitygradients/concat_grad/Slice_1'^gradients/concat_grad/tuple/group_deps*0
_class&
$"loc:@gradients/concat_grad/Slice_1*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
Dgradients/bidirectional_rnn/fw/fw/transpose_1_grad/InvertPermutationInvertPermutation bidirectional_rnn/fw/fw/concat_2*
T0*
_output_shapes
:

<gradients/bidirectional_rnn/fw/fw/transpose_1_grad/transpose	Transpose.gradients/concat_grad/tuple/control_dependencyDgradients/bidirectional_rnn/fw/fw/transpose_1_grad/InvertPermutation*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
Tperm0*
T0
ĺ
.gradients/ReverseSequence_grad/ReverseSequenceReverseSequence0gradients/concat_grad/tuple/control_dependency_1Placeholder_2*
T0*
	batch_dim *
seq_dim*

Tlen0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@
ş
mgradients/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3#bidirectional_rnn/fw/fw/TensorArray$bidirectional_rnn/fw/fw/while/Exit_2*6
_class,
*(loc:@bidirectional_rnn/fw/fw/TensorArray*
_output_shapes

:: *
source	gradients
ä
igradients/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flowIdentity$bidirectional_rnn/fw/fw/while/Exit_2n^gradients/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3*6
_class,
*(loc:@bidirectional_rnn/fw/fw/TensorArray*
T0*
_output_shapes
: 
ô
sgradients/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3mgradients/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3.bidirectional_rnn/fw/fw/TensorArrayStack/range<gradients/bidirectional_rnn/fw/fw/transpose_1_grad/transposeigradients/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flow*
T0*
_output_shapes
: 
p
gradients/zeros/shape_as_tensorConst*
valueB"   @   *
_output_shapes
:*
dtype0
Z
gradients/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    

gradients/zerosFillgradients/zeros/shape_as_tensorgradients/zeros/Const*
T0*
_output_shapes
:	@*

index_type0
r
!gradients/zeros_1/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"   @   
\
gradients/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    

gradients/zeros_1Fill!gradients/zeros_1/shape_as_tensorgradients/zeros_1/Const*

index_type0*
T0*
_output_shapes
:	@
 
Dgradients/bidirectional_rnn/bw/bw/transpose_1_grad/InvertPermutationInvertPermutation bidirectional_rnn/bw/bw/concat_2*
_output_shapes
:*
T0

<gradients/bidirectional_rnn/bw/bw/transpose_1_grad/transpose	Transpose.gradients/ReverseSequence_grad/ReverseSequenceDgradients/bidirectional_rnn/bw/bw/transpose_1_grad/InvertPermutation*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
Tperm0*
T0
Î
:gradients/bidirectional_rnn/fw/fw/while/Exit_2_grad/b_exitEntersgradients/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3*E

frame_name75gradients/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
: *
T0*
parallel_iterations *
is_constant( 
ó
:gradients/bidirectional_rnn/fw/fw/while/Exit_3_grad/b_exitEntergradients/zeros*
is_constant( *
_output_shapes
:	@*
parallel_iterations *
T0*E

frame_name75gradients/bidirectional_rnn/fw/fw/while/while_context
ő
:gradients/bidirectional_rnn/fw/fw/while/Exit_4_grad/b_exitEntergradients/zeros_1*
_output_shapes
:	@*
is_constant( *
T0*E

frame_name75gradients/bidirectional_rnn/fw/fw/while/while_context*
parallel_iterations 
ş
mgradients/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3#bidirectional_rnn/bw/bw/TensorArray$bidirectional_rnn/bw/bw/while/Exit_2*
_output_shapes

:: *6
_class,
*(loc:@bidirectional_rnn/bw/bw/TensorArray*
source	gradients
ä
igradients/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flowIdentity$bidirectional_rnn/bw/bw/while/Exit_2n^gradients/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*
_output_shapes
: *6
_class,
*(loc:@bidirectional_rnn/bw/bw/TensorArray
ô
sgradients/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3mgradients/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3.bidirectional_rnn/bw/bw/TensorArrayStack/range<gradients/bidirectional_rnn/bw/bw/transpose_1_grad/transposeigradients/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flow*
_output_shapes
: *
T0
r
!gradients/zeros_2/shape_as_tensorConst*
valueB"   @   *
dtype0*
_output_shapes
:
\
gradients/zeros_2/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    

gradients/zeros_2Fill!gradients/zeros_2/shape_as_tensorgradients/zeros_2/Const*

index_type0*
T0*
_output_shapes
:	@
r
!gradients/zeros_3/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"   @   
\
gradients/zeros_3/ConstConst*
dtype0*
valueB
 *    *
_output_shapes
: 

gradients/zeros_3Fill!gradients/zeros_3/shape_as_tensorgradients/zeros_3/Const*

index_type0*
_output_shapes
:	@*
T0
ö
>gradients/bidirectional_rnn/fw/fw/while/Switch_2_grad/b_switchMerge:gradients/bidirectional_rnn/fw/fw/while/Exit_2_grad/b_exitEgradients/bidirectional_rnn/fw/fw/while/Switch_2_grad_1/NextIteration*
T0*
_output_shapes
: : *
N
˙
>gradients/bidirectional_rnn/fw/fw/while/Switch_3_grad/b_switchMerge:gradients/bidirectional_rnn/fw/fw/while/Exit_3_grad/b_exitEgradients/bidirectional_rnn/fw/fw/while/Switch_3_grad_1/NextIteration*
N*
T0*!
_output_shapes
:	@: 
˙
>gradients/bidirectional_rnn/fw/fw/while/Switch_4_grad/b_switchMerge:gradients/bidirectional_rnn/fw/fw/while/Exit_4_grad/b_exitEgradients/bidirectional_rnn/fw/fw/while/Switch_4_grad_1/NextIteration*!
_output_shapes
:	@: *
N*
T0
Î
:gradients/bidirectional_rnn/bw/bw/while/Exit_2_grad/b_exitEntersgradients/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3*
_output_shapes
: *
T0*E

frame_name75gradients/bidirectional_rnn/bw/bw/while/while_context*
is_constant( *
parallel_iterations 
ő
:gradients/bidirectional_rnn/bw/bw/while/Exit_3_grad/b_exitEntergradients/zeros_2*
_output_shapes
:	@*
parallel_iterations *
T0*
is_constant( *E

frame_name75gradients/bidirectional_rnn/bw/bw/while/while_context
ő
:gradients/bidirectional_rnn/bw/bw/while/Exit_4_grad/b_exitEntergradients/zeros_3*
parallel_iterations *
_output_shapes
:	@*E

frame_name75gradients/bidirectional_rnn/bw/bw/while/while_context*
is_constant( *
T0

;gradients/bidirectional_rnn/fw/fw/while/Merge_2_grad/SwitchSwitch>gradients/bidirectional_rnn/fw/fw/while/Switch_2_grad/b_switchgradients/b_count_2*
_output_shapes
: : *Q
_classG
ECloc:@gradients/bidirectional_rnn/fw/fw/while/Switch_2_grad/b_switch*
T0

Egradients/bidirectional_rnn/fw/fw/while/Merge_2_grad/tuple/group_depsNoOp<^gradients/bidirectional_rnn/fw/fw/while/Merge_2_grad/Switch
Ň
Mgradients/bidirectional_rnn/fw/fw/while/Merge_2_grad/tuple/control_dependencyIdentity;gradients/bidirectional_rnn/fw/fw/while/Merge_2_grad/SwitchF^gradients/bidirectional_rnn/fw/fw/while/Merge_2_grad/tuple/group_deps*
T0*
_output_shapes
: *Q
_classG
ECloc:@gradients/bidirectional_rnn/fw/fw/while/Switch_2_grad/b_switch
Ö
Ogradients/bidirectional_rnn/fw/fw/while/Merge_2_grad/tuple/control_dependency_1Identity=gradients/bidirectional_rnn/fw/fw/while/Merge_2_grad/Switch:1F^gradients/bidirectional_rnn/fw/fw/while/Merge_2_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/bidirectional_rnn/fw/fw/while/Switch_2_grad/b_switch*
_output_shapes
: 
˘
;gradients/bidirectional_rnn/fw/fw/while/Merge_3_grad/SwitchSwitch>gradients/bidirectional_rnn/fw/fw/while/Switch_3_grad/b_switchgradients/b_count_2**
_output_shapes
:	@:	@*Q
_classG
ECloc:@gradients/bidirectional_rnn/fw/fw/while/Switch_3_grad/b_switch*
T0

Egradients/bidirectional_rnn/fw/fw/while/Merge_3_grad/tuple/group_depsNoOp<^gradients/bidirectional_rnn/fw/fw/while/Merge_3_grad/Switch
Ű
Mgradients/bidirectional_rnn/fw/fw/while/Merge_3_grad/tuple/control_dependencyIdentity;gradients/bidirectional_rnn/fw/fw/while/Merge_3_grad/SwitchF^gradients/bidirectional_rnn/fw/fw/while/Merge_3_grad/tuple/group_deps*Q
_classG
ECloc:@gradients/bidirectional_rnn/fw/fw/while/Switch_3_grad/b_switch*
_output_shapes
:	@*
T0
ß
Ogradients/bidirectional_rnn/fw/fw/while/Merge_3_grad/tuple/control_dependency_1Identity=gradients/bidirectional_rnn/fw/fw/while/Merge_3_grad/Switch:1F^gradients/bidirectional_rnn/fw/fw/while/Merge_3_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/bidirectional_rnn/fw/fw/while/Switch_3_grad/b_switch*
_output_shapes
:	@
˘
;gradients/bidirectional_rnn/fw/fw/while/Merge_4_grad/SwitchSwitch>gradients/bidirectional_rnn/fw/fw/while/Switch_4_grad/b_switchgradients/b_count_2*
T0*Q
_classG
ECloc:@gradients/bidirectional_rnn/fw/fw/while/Switch_4_grad/b_switch**
_output_shapes
:	@:	@

Egradients/bidirectional_rnn/fw/fw/while/Merge_4_grad/tuple/group_depsNoOp<^gradients/bidirectional_rnn/fw/fw/while/Merge_4_grad/Switch
Ű
Mgradients/bidirectional_rnn/fw/fw/while/Merge_4_grad/tuple/control_dependencyIdentity;gradients/bidirectional_rnn/fw/fw/while/Merge_4_grad/SwitchF^gradients/bidirectional_rnn/fw/fw/while/Merge_4_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/bidirectional_rnn/fw/fw/while/Switch_4_grad/b_switch*
_output_shapes
:	@
ß
Ogradients/bidirectional_rnn/fw/fw/while/Merge_4_grad/tuple/control_dependency_1Identity=gradients/bidirectional_rnn/fw/fw/while/Merge_4_grad/Switch:1F^gradients/bidirectional_rnn/fw/fw/while/Merge_4_grad/tuple/group_deps*
_output_shapes
:	@*Q
_classG
ECloc:@gradients/bidirectional_rnn/fw/fw/while/Switch_4_grad/b_switch*
T0
ö
>gradients/bidirectional_rnn/bw/bw/while/Switch_2_grad/b_switchMerge:gradients/bidirectional_rnn/bw/bw/while/Exit_2_grad/b_exitEgradients/bidirectional_rnn/bw/bw/while/Switch_2_grad_1/NextIteration*
_output_shapes
: : *
N*
T0
˙
>gradients/bidirectional_rnn/bw/bw/while/Switch_3_grad/b_switchMerge:gradients/bidirectional_rnn/bw/bw/while/Exit_3_grad/b_exitEgradients/bidirectional_rnn/bw/bw/while/Switch_3_grad_1/NextIteration*
N*
T0*!
_output_shapes
:	@: 
˙
>gradients/bidirectional_rnn/bw/bw/while/Switch_4_grad/b_switchMerge:gradients/bidirectional_rnn/bw/bw/while/Exit_4_grad/b_exitEgradients/bidirectional_rnn/bw/bw/while/Switch_4_grad_1/NextIteration*
N*
T0*!
_output_shapes
:	@: 
ą
9gradients/bidirectional_rnn/fw/fw/while/Enter_2_grad/ExitExitMgradients/bidirectional_rnn/fw/fw/while/Merge_2_grad/tuple/control_dependency*
_output_shapes
: *
T0
ş
9gradients/bidirectional_rnn/fw/fw/while/Enter_3_grad/ExitExitMgradients/bidirectional_rnn/fw/fw/while/Merge_3_grad/tuple/control_dependency*
T0*
_output_shapes
:	@
ş
9gradients/bidirectional_rnn/fw/fw/while/Enter_4_grad/ExitExitMgradients/bidirectional_rnn/fw/fw/while/Merge_4_grad/tuple/control_dependency*
_output_shapes
:	@*
T0

;gradients/bidirectional_rnn/bw/bw/while/Merge_2_grad/SwitchSwitch>gradients/bidirectional_rnn/bw/bw/while/Switch_2_grad/b_switchgradients/b_count_6*
T0*Q
_classG
ECloc:@gradients/bidirectional_rnn/bw/bw/while/Switch_2_grad/b_switch*
_output_shapes
: : 

Egradients/bidirectional_rnn/bw/bw/while/Merge_2_grad/tuple/group_depsNoOp<^gradients/bidirectional_rnn/bw/bw/while/Merge_2_grad/Switch
Ň
Mgradients/bidirectional_rnn/bw/bw/while/Merge_2_grad/tuple/control_dependencyIdentity;gradients/bidirectional_rnn/bw/bw/while/Merge_2_grad/SwitchF^gradients/bidirectional_rnn/bw/bw/while/Merge_2_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/bidirectional_rnn/bw/bw/while/Switch_2_grad/b_switch*
_output_shapes
: 
Ö
Ogradients/bidirectional_rnn/bw/bw/while/Merge_2_grad/tuple/control_dependency_1Identity=gradients/bidirectional_rnn/bw/bw/while/Merge_2_grad/Switch:1F^gradients/bidirectional_rnn/bw/bw/while/Merge_2_grad/tuple/group_deps*Q
_classG
ECloc:@gradients/bidirectional_rnn/bw/bw/while/Switch_2_grad/b_switch*
_output_shapes
: *
T0
˘
;gradients/bidirectional_rnn/bw/bw/while/Merge_3_grad/SwitchSwitch>gradients/bidirectional_rnn/bw/bw/while/Switch_3_grad/b_switchgradients/b_count_6*Q
_classG
ECloc:@gradients/bidirectional_rnn/bw/bw/while/Switch_3_grad/b_switch*
T0**
_output_shapes
:	@:	@

Egradients/bidirectional_rnn/bw/bw/while/Merge_3_grad/tuple/group_depsNoOp<^gradients/bidirectional_rnn/bw/bw/while/Merge_3_grad/Switch
Ű
Mgradients/bidirectional_rnn/bw/bw/while/Merge_3_grad/tuple/control_dependencyIdentity;gradients/bidirectional_rnn/bw/bw/while/Merge_3_grad/SwitchF^gradients/bidirectional_rnn/bw/bw/while/Merge_3_grad/tuple/group_deps*
_output_shapes
:	@*Q
_classG
ECloc:@gradients/bidirectional_rnn/bw/bw/while/Switch_3_grad/b_switch*
T0
ß
Ogradients/bidirectional_rnn/bw/bw/while/Merge_3_grad/tuple/control_dependency_1Identity=gradients/bidirectional_rnn/bw/bw/while/Merge_3_grad/Switch:1F^gradients/bidirectional_rnn/bw/bw/while/Merge_3_grad/tuple/group_deps*Q
_classG
ECloc:@gradients/bidirectional_rnn/bw/bw/while/Switch_3_grad/b_switch*
_output_shapes
:	@*
T0
˘
;gradients/bidirectional_rnn/bw/bw/while/Merge_4_grad/SwitchSwitch>gradients/bidirectional_rnn/bw/bw/while/Switch_4_grad/b_switchgradients/b_count_6**
_output_shapes
:	@:	@*
T0*Q
_classG
ECloc:@gradients/bidirectional_rnn/bw/bw/while/Switch_4_grad/b_switch

Egradients/bidirectional_rnn/bw/bw/while/Merge_4_grad/tuple/group_depsNoOp<^gradients/bidirectional_rnn/bw/bw/while/Merge_4_grad/Switch
Ű
Mgradients/bidirectional_rnn/bw/bw/while/Merge_4_grad/tuple/control_dependencyIdentity;gradients/bidirectional_rnn/bw/bw/while/Merge_4_grad/SwitchF^gradients/bidirectional_rnn/bw/bw/while/Merge_4_grad/tuple/group_deps*
T0*
_output_shapes
:	@*Q
_classG
ECloc:@gradients/bidirectional_rnn/bw/bw/while/Switch_4_grad/b_switch
ß
Ogradients/bidirectional_rnn/bw/bw/while/Merge_4_grad/tuple/control_dependency_1Identity=gradients/bidirectional_rnn/bw/bw/while/Merge_4_grad/Switch:1F^gradients/bidirectional_rnn/bw/bw/while/Merge_4_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/bidirectional_rnn/bw/bw/while/Switch_4_grad/b_switch*
_output_shapes
:	@
Ç
rgradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3xgradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterOgradients/bidirectional_rnn/fw/fw/while/Merge_2_grad/tuple/control_dependency_1*
source	gradients*
_output_shapes

:: *>
_class4
20loc:@bidirectional_rnn/fw/fw/while/dropout/mul_1

xgradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnter#bidirectional_rnn/fw/fw/TensorArray*
T0*
parallel_iterations *
_output_shapes
:*E

frame_name75gradients/bidirectional_rnn/fw/fw/while/while_context*>
_class4
20loc:@bidirectional_rnn/fw/fw/while/dropout/mul_1*
is_constant(
Ą
ngradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flowIdentityOgradients/bidirectional_rnn/fw/fw/while/Merge_2_grad/tuple/control_dependency_1s^gradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*
_output_shapes
: *>
_class4
20loc:@bidirectional_rnn/fw/fw/while/dropout/mul_1
ř
bgradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3TensorArrayReadV3rgradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3mgradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2ngradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flow*
dtype0*
_output_shapes
:	@
đ
hgradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/ConstConst*
valueB :
˙˙˙˙˙˙˙˙˙*;
_class1
/-loc:@bidirectional_rnn/fw/fw/while/Identity_1*
dtype0*
_output_shapes
: 
Ů
hgradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_accStackV2hgradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Const*

stack_name *;
_class1
/-loc:@bidirectional_rnn/fw/fw/while/Identity_1*
_output_shapes
:*
	elem_type0
ë
hgradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/EnterEnterhgradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
_output_shapes
:*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context*
T0*
parallel_iterations *
is_constant(
Ő
ngradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2StackPushV2hgradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter(bidirectional_rnn/fw/fw/while/Identity_1^gradients/Add*
_output_shapes
: *
swap_memory( *
T0
Š
mgradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
StackPopV2sgradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/Enter^gradients/Sub*
_output_shapes
: *
	elem_type0

sgradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/EnterEnterhgradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
_output_shapes
:*E

frame_name75gradients/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
parallel_iterations 
š	
igradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_syncControlTriggerH^gradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2n^gradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2X^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2R^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2T^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2R^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2T^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2P^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2R^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2J^gradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul/StackPopV2L^gradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul_1/StackPopV2H^gradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/StackPopV2J^gradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/StackPopV2
 
agradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_depsNoOpP^gradients/bidirectional_rnn/fw/fw/while/Merge_2_grad/tuple/control_dependency_1c^gradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3
Ţ
igradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependencyIdentitybgradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3b^gradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*u
_classk
igloc:@gradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3*
T0*
_output_shapes
:	@
 
kgradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1IdentityOgradients/bidirectional_rnn/fw/fw/while/Merge_2_grad/tuple/control_dependency_1b^gradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*Q
_classG
ECloc:@gradients/bidirectional_rnn/fw/fw/while/Switch_2_grad/b_switch*
_output_shapes
: *
T0
ą
Pgradients/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/shape_as_tensorConst^gradients/Sub*
valueB"   @   *
_output_shapes
:*
dtype0

Fgradients/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/ConstConst^gradients/Sub*
dtype0*
_output_shapes
: *
valueB
 *    

@gradients/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_likeFillPgradients/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/shape_as_tensorFgradients/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/Const*
T0*

index_type0*
_output_shapes
:	@
Ě
<gradients/bidirectional_rnn/fw/fw/while/Select_1_grad/SelectSelectGgradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2Ogradients/bidirectional_rnn/fw/fw/while/Merge_3_grad/tuple/control_dependency_1@gradients/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like*
T0*
_output_shapes
:	@
Ě
Bgradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/ConstConst*
valueB :
˙˙˙˙˙˙˙˙˙*
_output_shapes
: *=
_class3
1/loc:@bidirectional_rnn/fw/fw/while/GreaterEqual*
dtype0

Bgradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/f_accStackV2Bgradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/Const*
	elem_type0
*

stack_name *
_output_shapes
:*=
_class3
1/loc:@bidirectional_rnn/fw/fw/while/GreaterEqual

Bgradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/EnterEnterBgradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/f_acc*
_output_shapes
:*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
T0*
parallel_iterations 

Hgradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPushV2StackPushV2Bgradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/Enter*bidirectional_rnn/fw/fw/while/GreaterEqual^gradients/Add*
swap_memory( *
T0
*
_output_shapes
:
ß
Ggradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2
StackPopV2Mgradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2/Enter^gradients/Sub*
	elem_type0
*
_output_shapes
:
´
Mgradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2/EnterEnterBgradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/f_acc*
_output_shapes
:*
parallel_iterations *E

frame_name75gradients/bidirectional_rnn/fw/fw/while/while_context*
T0*
is_constant(
Î
>gradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select_1SelectGgradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2@gradients/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_likeOgradients/bidirectional_rnn/fw/fw/while/Merge_3_grad/tuple/control_dependency_1*
T0*
_output_shapes
:	@
Î
Fgradients/bidirectional_rnn/fw/fw/while/Select_1_grad/tuple/group_depsNoOp=^gradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select?^gradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select_1
Ü
Ngradients/bidirectional_rnn/fw/fw/while/Select_1_grad/tuple/control_dependencyIdentity<gradients/bidirectional_rnn/fw/fw/while/Select_1_grad/SelectG^gradients/bidirectional_rnn/fw/fw/while/Select_1_grad/tuple/group_deps*
_output_shapes
:	@*
T0*O
_classE
CAloc:@gradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select
â
Pgradients/bidirectional_rnn/fw/fw/while/Select_1_grad/tuple/control_dependency_1Identity>gradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select_1G^gradients/bidirectional_rnn/fw/fw/while/Select_1_grad/tuple/group_deps*
T0*
_output_shapes
:	@*Q
_classG
ECloc:@gradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select_1
ą
Pgradients/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/shape_as_tensorConst^gradients/Sub*
valueB"   @   *
_output_shapes
:*
dtype0

Fgradients/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/ConstConst^gradients/Sub*
dtype0*
_output_shapes
: *
valueB
 *    

@gradients/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_likeFillPgradients/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/shape_as_tensorFgradients/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/Const*
T0*
_output_shapes
:	@*

index_type0
Ě
<gradients/bidirectional_rnn/fw/fw/while/Select_2_grad/SelectSelectGgradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2Ogradients/bidirectional_rnn/fw/fw/while/Merge_4_grad/tuple/control_dependency_1@gradients/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like*
_output_shapes
:	@*
T0
Î
>gradients/bidirectional_rnn/fw/fw/while/Select_2_grad/Select_1SelectGgradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2@gradients/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_likeOgradients/bidirectional_rnn/fw/fw/while/Merge_4_grad/tuple/control_dependency_1*
T0*
_output_shapes
:	@
Î
Fgradients/bidirectional_rnn/fw/fw/while/Select_2_grad/tuple/group_depsNoOp=^gradients/bidirectional_rnn/fw/fw/while/Select_2_grad/Select?^gradients/bidirectional_rnn/fw/fw/while/Select_2_grad/Select_1
Ü
Ngradients/bidirectional_rnn/fw/fw/while/Select_2_grad/tuple/control_dependencyIdentity<gradients/bidirectional_rnn/fw/fw/while/Select_2_grad/SelectG^gradients/bidirectional_rnn/fw/fw/while/Select_2_grad/tuple/group_deps*O
_classE
CAloc:@gradients/bidirectional_rnn/fw/fw/while/Select_2_grad/Select*
_output_shapes
:	@*
T0
â
Pgradients/bidirectional_rnn/fw/fw/while/Select_2_grad/tuple/control_dependency_1Identity>gradients/bidirectional_rnn/fw/fw/while/Select_2_grad/Select_1G^gradients/bidirectional_rnn/fw/fw/while/Select_2_grad/tuple/group_deps*Q
_classG
ECloc:@gradients/bidirectional_rnn/fw/fw/while/Select_2_grad/Select_1*
_output_shapes
:	@*
T0
ą
9gradients/bidirectional_rnn/bw/bw/while/Enter_2_grad/ExitExitMgradients/bidirectional_rnn/bw/bw/while/Merge_2_grad/tuple/control_dependency*
_output_shapes
: *
T0
ş
9gradients/bidirectional_rnn/bw/bw/while/Enter_3_grad/ExitExitMgradients/bidirectional_rnn/bw/bw/while/Merge_3_grad/tuple/control_dependency*
_output_shapes
:	@*
T0
ş
9gradients/bidirectional_rnn/bw/bw/while/Enter_4_grad/ExitExitMgradients/bidirectional_rnn/bw/bw/while/Merge_4_grad/tuple/control_dependency*
T0*
_output_shapes
:	@
Ż
Ngradients/bidirectional_rnn/fw/fw/while/Select_grad/zeros_like/shape_as_tensorConst^gradients/Sub*
dtype0*
valueB"   @   *
_output_shapes
:

Dgradients/bidirectional_rnn/fw/fw/while/Select_grad/zeros_like/ConstConst^gradients/Sub*
dtype0*
valueB
 *    *
_output_shapes
: 

>gradients/bidirectional_rnn/fw/fw/while/Select_grad/zeros_likeFillNgradients/bidirectional_rnn/fw/fw/while/Select_grad/zeros_like/shape_as_tensorDgradients/bidirectional_rnn/fw/fw/while/Select_grad/zeros_like/Const*

index_type0*
_output_shapes
:	@*
T0
â
:gradients/bidirectional_rnn/fw/fw/while/Select_grad/SelectSelectGgradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2igradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency>gradients/bidirectional_rnn/fw/fw/while/Select_grad/zeros_like*
_output_shapes
:	@*
T0
ä
<gradients/bidirectional_rnn/fw/fw/while/Select_grad/Select_1SelectGgradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2>gradients/bidirectional_rnn/fw/fw/while/Select_grad/zeros_likeigradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency*
_output_shapes
:	@*
T0
Č
Dgradients/bidirectional_rnn/fw/fw/while/Select_grad/tuple/group_depsNoOp;^gradients/bidirectional_rnn/fw/fw/while/Select_grad/Select=^gradients/bidirectional_rnn/fw/fw/while/Select_grad/Select_1
Ô
Lgradients/bidirectional_rnn/fw/fw/while/Select_grad/tuple/control_dependencyIdentity:gradients/bidirectional_rnn/fw/fw/while/Select_grad/SelectE^gradients/bidirectional_rnn/fw/fw/while/Select_grad/tuple/group_deps*
_output_shapes
:	@*
T0*M
_classC
A?loc:@gradients/bidirectional_rnn/fw/fw/while/Select_grad/Select
Ú
Ngradients/bidirectional_rnn/fw/fw/while/Select_grad/tuple/control_dependency_1Identity<gradients/bidirectional_rnn/fw/fw/while/Select_grad/Select_1E^gradients/bidirectional_rnn/fw/fw/while/Select_grad/tuple/group_deps*
T0*
_output_shapes
:	@*O
_classE
CAloc:@gradients/bidirectional_rnn/fw/fw/while/Select_grad/Select_1
Ç
rgradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3xgradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterOgradients/bidirectional_rnn/bw/bw/while/Merge_2_grad/tuple/control_dependency_1*>
_class4
20loc:@bidirectional_rnn/bw/bw/while/dropout/mul_1*
_output_shapes

:: *
source	gradients

xgradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnter#bidirectional_rnn/bw/bw/TensorArray*
parallel_iterations *
T0*
_output_shapes
:*>
_class4
20loc:@bidirectional_rnn/bw/bw/while/dropout/mul_1*
is_constant(*E

frame_name75gradients/bidirectional_rnn/bw/bw/while/while_context
Ą
ngradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flowIdentityOgradients/bidirectional_rnn/bw/bw/while/Merge_2_grad/tuple/control_dependency_1s^gradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3*>
_class4
20loc:@bidirectional_rnn/bw/bw/while/dropout/mul_1*
_output_shapes
: *
T0
ř
bgradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3TensorArrayReadV3rgradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3mgradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2ngradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flow*
_output_shapes
:	@*
dtype0
đ
hgradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/ConstConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙*;
_class1
/-loc:@bidirectional_rnn/bw/bw/while/Identity_1
Ů
hgradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_accStackV2hgradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Const*

stack_name *
	elem_type0*;
_class1
/-loc:@bidirectional_rnn/bw/bw/while/Identity_1*
_output_shapes
:
ë
hgradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/EnterEnterhgradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
_output_shapes
:*
parallel_iterations *;

frame_name-+bidirectional_rnn/bw/bw/while/while_context*
T0*
is_constant(
×
ngradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2StackPushV2hgradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter(bidirectional_rnn/bw/bw/while/Identity_1^gradients/Add_1*
_output_shapes
: *
T0*
swap_memory( 
Ť
mgradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
StackPopV2sgradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/Enter^gradients/Sub_1*
	elem_type0*
_output_shapes
: 

sgradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/EnterEnterhgradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
_output_shapes
:*
is_constant(*E

frame_name75gradients/bidirectional_rnn/bw/bw/while/while_context*
parallel_iterations 
š	
igradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_syncControlTriggerH^gradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2n^gradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2X^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2R^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2T^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2R^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2T^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2P^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2R^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2J^gradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul/StackPopV2L^gradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul_1/StackPopV2H^gradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/StackPopV2J^gradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/StackPopV2
 
agradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_depsNoOpP^gradients/bidirectional_rnn/bw/bw/while/Merge_2_grad/tuple/control_dependency_1c^gradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3
Ţ
igradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependencyIdentitybgradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3b^gradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*u
_classk
igloc:@gradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3*
_output_shapes
:	@*
T0
 
kgradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1IdentityOgradients/bidirectional_rnn/bw/bw/while/Merge_2_grad/tuple/control_dependency_1b^gradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/bidirectional_rnn/bw/bw/while/Switch_2_grad/b_switch*
_output_shapes
: 
ł
Pgradients/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/shape_as_tensorConst^gradients/Sub_1*
valueB"   @   *
_output_shapes
:*
dtype0

Fgradients/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/ConstConst^gradients/Sub_1*
dtype0*
valueB
 *    *
_output_shapes
: 

@gradients/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_likeFillPgradients/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/shape_as_tensorFgradients/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/Const*

index_type0*
_output_shapes
:	@*
T0
Ě
<gradients/bidirectional_rnn/bw/bw/while/Select_1_grad/SelectSelectGgradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2Ogradients/bidirectional_rnn/bw/bw/while/Merge_3_grad/tuple/control_dependency_1@gradients/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like*
T0*
_output_shapes
:	@
Ě
Bgradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/ConstConst*=
_class3
1/loc:@bidirectional_rnn/bw/bw/while/GreaterEqual*
dtype0*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙

Bgradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/f_accStackV2Bgradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/Const*

stack_name *=
_class3
1/loc:@bidirectional_rnn/bw/bw/while/GreaterEqual*
_output_shapes
:*
	elem_type0


Bgradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/EnterEnterBgradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/f_acc*
is_constant(*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:*
T0*
parallel_iterations 

Hgradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPushV2StackPushV2Bgradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/Enter*bidirectional_rnn/bw/bw/while/GreaterEqual^gradients/Add_1*
T0
*
swap_memory( *
_output_shapes
:
á
Ggradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2
StackPopV2Mgradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2/Enter^gradients/Sub_1*
	elem_type0
*
_output_shapes
:
´
Mgradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2/EnterEnterBgradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/f_acc*E

frame_name75gradients/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:*
is_constant(*
T0*
parallel_iterations 
Î
>gradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select_1SelectGgradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2@gradients/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_likeOgradients/bidirectional_rnn/bw/bw/while/Merge_3_grad/tuple/control_dependency_1*
T0*
_output_shapes
:	@
Î
Fgradients/bidirectional_rnn/bw/bw/while/Select_1_grad/tuple/group_depsNoOp=^gradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select?^gradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select_1
Ü
Ngradients/bidirectional_rnn/bw/bw/while/Select_1_grad/tuple/control_dependencyIdentity<gradients/bidirectional_rnn/bw/bw/while/Select_1_grad/SelectG^gradients/bidirectional_rnn/bw/bw/while/Select_1_grad/tuple/group_deps*
_output_shapes
:	@*
T0*O
_classE
CAloc:@gradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select
â
Pgradients/bidirectional_rnn/bw/bw/while/Select_1_grad/tuple/control_dependency_1Identity>gradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select_1G^gradients/bidirectional_rnn/bw/bw/while/Select_1_grad/tuple/group_deps*
_output_shapes
:	@*Q
_classG
ECloc:@gradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select_1*
T0
ł
Pgradients/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/shape_as_tensorConst^gradients/Sub_1*
_output_shapes
:*
dtype0*
valueB"   @   

Fgradients/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/ConstConst^gradients/Sub_1*
valueB
 *    *
dtype0*
_output_shapes
: 

@gradients/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_likeFillPgradients/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/shape_as_tensorFgradients/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/Const*
T0*

index_type0*
_output_shapes
:	@
Ě
<gradients/bidirectional_rnn/bw/bw/while/Select_2_grad/SelectSelectGgradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2Ogradients/bidirectional_rnn/bw/bw/while/Merge_4_grad/tuple/control_dependency_1@gradients/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like*
T0*
_output_shapes
:	@
Î
>gradients/bidirectional_rnn/bw/bw/while/Select_2_grad/Select_1SelectGgradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2@gradients/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_likeOgradients/bidirectional_rnn/bw/bw/while/Merge_4_grad/tuple/control_dependency_1*
_output_shapes
:	@*
T0
Î
Fgradients/bidirectional_rnn/bw/bw/while/Select_2_grad/tuple/group_depsNoOp=^gradients/bidirectional_rnn/bw/bw/while/Select_2_grad/Select?^gradients/bidirectional_rnn/bw/bw/while/Select_2_grad/Select_1
Ü
Ngradients/bidirectional_rnn/bw/bw/while/Select_2_grad/tuple/control_dependencyIdentity<gradients/bidirectional_rnn/bw/bw/while/Select_2_grad/SelectG^gradients/bidirectional_rnn/bw/bw/while/Select_2_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/bidirectional_rnn/bw/bw/while/Select_2_grad/Select*
_output_shapes
:	@
â
Pgradients/bidirectional_rnn/bw/bw/while/Select_2_grad/tuple/control_dependency_1Identity>gradients/bidirectional_rnn/bw/bw/while/Select_2_grad/Select_1G^gradients/bidirectional_rnn/bw/bw/while/Select_2_grad/tuple/group_deps*
_output_shapes
:	@*
T0*Q
_classG
ECloc:@gradients/bidirectional_rnn/bw/bw/while/Select_2_grad/Select_1

>gradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/MulMulNgradients/bidirectional_rnn/fw/fw/while/Select_grad/tuple/control_dependency_1Igradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul/StackPopV2*
T0*
_output_shapes
:	@
Î
Dgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul/ConstConst*
valueB :
˙˙˙˙˙˙˙˙˙*
_output_shapes
: *=
_class3
1/loc:@bidirectional_rnn/fw/fw/while/dropout/Cast*
dtype0

Dgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul/f_accStackV2Dgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul/Const*
	elem_type0*=
_class3
1/loc:@bidirectional_rnn/fw/fw/while/dropout/Cast*

stack_name *
_output_shapes
:
Ł
Dgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul/EnterEnterDgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul/f_acc*
_output_shapes
:*
parallel_iterations *;

frame_name-+bidirectional_rnn/fw/fw/while/while_context*
T0*
is_constant(

Jgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul/StackPushV2StackPushV2Dgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul/Enter*bidirectional_rnn/fw/fw/while/dropout/Cast^gradients/Add*
T0*
_output_shapes
:	@*
swap_memory( 
ę
Igradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul/StackPopV2
StackPopV2Ogradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul/StackPopV2/Enter^gradients/Sub*
_output_shapes
:	@*
	elem_type0
¸
Ogradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul/StackPopV2/EnterEnterDgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul/f_acc*
is_constant(*
_output_shapes
:*
parallel_iterations *E

frame_name75gradients/bidirectional_rnn/fw/fw/while/while_context*
T0

@gradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul_1MulNgradients/bidirectional_rnn/fw/fw/while/Select_grad/tuple/control_dependency_1Kgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul_1/StackPopV2*
_output_shapes
:	@*
T0
Ď
Fgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul_1/ConstConst*
dtype0*
_output_shapes
: *<
_class2
0.loc:@bidirectional_rnn/fw/fw/while/dropout/mul*
valueB :
˙˙˙˙˙˙˙˙˙

Fgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul_1/f_accStackV2Fgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul_1/Const*

stack_name *
_output_shapes
:*<
_class2
0.loc:@bidirectional_rnn/fw/fw/while/dropout/mul*
	elem_type0
§
Fgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul_1/EnterEnterFgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul_1/f_acc*
T0*
_output_shapes
:*
parallel_iterations *
is_constant(*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context

Lgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul_1/StackPushV2StackPushV2Fgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul_1/Enter)bidirectional_rnn/fw/fw/while/dropout/mul^gradients/Add*
T0*
_output_shapes
:	@*
swap_memory( 
î
Kgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul_1/StackPopV2
StackPopV2Qgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul_1/StackPopV2/Enter^gradients/Sub*
_output_shapes
:	@*
	elem_type0
ź
Qgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul_1/StackPopV2/EnterEnterFgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul_1/f_acc*
T0*
_output_shapes
:*
is_constant(*E

frame_name75gradients/bidirectional_rnn/fw/fw/while/while_context*
parallel_iterations 
×
Kgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/tuple/group_depsNoOp?^gradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/MulA^gradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul_1
ę
Sgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/tuple/control_dependencyIdentity>gradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/MulL^gradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/tuple/group_deps*Q
_classG
ECloc:@gradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul*
_output_shapes
:	@*
T0
đ
Ugradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/tuple/control_dependency_1Identity@gradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul_1L^gradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/tuple/group_deps*
_output_shapes
:	@*
T0*S
_classI
GEloc:@gradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul_1
ä
Egradients/bidirectional_rnn/fw/fw/while/Switch_2_grad_1/NextIterationNextIterationkgradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1*
_output_shapes
: *
T0
ą
Ngradients/bidirectional_rnn/bw/bw/while/Select_grad/zeros_like/shape_as_tensorConst^gradients/Sub_1*
valueB"   @   *
_output_shapes
:*
dtype0

Dgradients/bidirectional_rnn/bw/bw/while/Select_grad/zeros_like/ConstConst^gradients/Sub_1*
valueB
 *    *
_output_shapes
: *
dtype0

>gradients/bidirectional_rnn/bw/bw/while/Select_grad/zeros_likeFillNgradients/bidirectional_rnn/bw/bw/while/Select_grad/zeros_like/shape_as_tensorDgradients/bidirectional_rnn/bw/bw/while/Select_grad/zeros_like/Const*
T0*
_output_shapes
:	@*

index_type0
â
:gradients/bidirectional_rnn/bw/bw/while/Select_grad/SelectSelectGgradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2igradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency>gradients/bidirectional_rnn/bw/bw/while/Select_grad/zeros_like*
T0*
_output_shapes
:	@
ä
<gradients/bidirectional_rnn/bw/bw/while/Select_grad/Select_1SelectGgradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2>gradients/bidirectional_rnn/bw/bw/while/Select_grad/zeros_likeigradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency*
_output_shapes
:	@*
T0
Č
Dgradients/bidirectional_rnn/bw/bw/while/Select_grad/tuple/group_depsNoOp;^gradients/bidirectional_rnn/bw/bw/while/Select_grad/Select=^gradients/bidirectional_rnn/bw/bw/while/Select_grad/Select_1
Ô
Lgradients/bidirectional_rnn/bw/bw/while/Select_grad/tuple/control_dependencyIdentity:gradients/bidirectional_rnn/bw/bw/while/Select_grad/SelectE^gradients/bidirectional_rnn/bw/bw/while/Select_grad/tuple/group_deps*M
_classC
A?loc:@gradients/bidirectional_rnn/bw/bw/while/Select_grad/Select*
_output_shapes
:	@*
T0
Ú
Ngradients/bidirectional_rnn/bw/bw/while/Select_grad/tuple/control_dependency_1Identity<gradients/bidirectional_rnn/bw/bw/while/Select_grad/Select_1E^gradients/bidirectional_rnn/bw/bw/while/Select_grad/tuple/group_deps*O
_classE
CAloc:@gradients/bidirectional_rnn/bw/bw/while/Select_grad/Select_1*
_output_shapes
:	@*
T0

>gradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/ShapeConst^gradients/Sub*
valueB"   @   *
_output_shapes
:*
dtype0

@gradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Shape_1Const^gradients/Sub*
_output_shapes
: *
valueB *
dtype0
Ś
Ngradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs>gradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Shape@gradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

<gradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/MulMulSgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/tuple/control_dependencyGgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/StackPopV2*
_output_shapes
:	@*
T0
Ď
Bgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/ConstConst*
valueB :
˙˙˙˙˙˙˙˙˙*@
_class6
42loc:@bidirectional_rnn/fw/fw/while/dropout/truediv*
_output_shapes
: *
dtype0

Bgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/f_accStackV2Bgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/Const*
	elem_type0*

stack_name *@
_class6
42loc:@bidirectional_rnn/fw/fw/while/dropout/truediv*
_output_shapes
:

Bgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/EnterEnterBgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/f_acc*
_output_shapes
:*
parallel_iterations *;

frame_name-+bidirectional_rnn/fw/fw/while/while_context*
T0*
is_constant(

Hgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/StackPushV2StackPushV2Bgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/Enter-bidirectional_rnn/fw/fw/while/dropout/truediv^gradients/Add*
swap_memory( *
_output_shapes
: *
T0
Ý
Ggradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/StackPopV2
StackPopV2Mgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
: 
´
Mgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/StackPopV2/EnterEnterBgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/f_acc*
T0*
_output_shapes
:*
parallel_iterations *E

frame_name75gradients/bidirectional_rnn/fw/fw/while/while_context*
is_constant(

<gradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/SumSum<gradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/MulNgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs*
_output_shapes
:	@*

Tidx0*
T0*
	keep_dims( 

@gradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/ReshapeReshape<gradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Sum>gradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
:	@

>gradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1MulIgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/StackPopV2Sgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/tuple/control_dependency*
T0*
_output_shapes
:	@
×
Dgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/ConstConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙*F
_class<
:8loc:@bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2

Dgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/f_accStackV2Dgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/Const*
_output_shapes
:*F
_class<
:8loc:@bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2*

stack_name *
	elem_type0
Ł
Dgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/EnterEnterDgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/f_acc*
is_constant(*
_output_shapes
:*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context*
parallel_iterations *
T0
Ą
Jgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/StackPushV2StackPushV2Dgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/Enter3bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2^gradients/Add*
_output_shapes
:	@*
swap_memory( *
T0
ę
Igradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/StackPopV2
StackPopV2Ogradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/StackPopV2/Enter^gradients/Sub*
_output_shapes
:	@*
	elem_type0
¸
Ogradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/StackPopV2/EnterEnterDgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/f_acc*
is_constant(*
parallel_iterations *
_output_shapes
:*E

frame_name75gradients/bidirectional_rnn/fw/fw/while/while_context*
T0

>gradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Sum_1Sum>gradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1Pgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs:1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
ţ
Bgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Reshape_1Reshape>gradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Sum_1@gradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Ů
Igradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/tuple/group_depsNoOpA^gradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/ReshapeC^gradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Reshape_1
ę
Qgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/tuple/control_dependencyIdentity@gradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/ReshapeJ^gradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Reshape*
_output_shapes
:	@
ç
Sgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/tuple/control_dependency_1IdentityBgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Reshape_1J^gradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Reshape_1*
_output_shapes
: 

>gradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/MulMulNgradients/bidirectional_rnn/bw/bw/while/Select_grad/tuple/control_dependency_1Igradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul/StackPopV2*
_output_shapes
:	@*
T0
Î
Dgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul/ConstConst*
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙*
_output_shapes
: *=
_class3
1/loc:@bidirectional_rnn/bw/bw/while/dropout/Cast

Dgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul/f_accStackV2Dgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul/Const*

stack_name *=
_class3
1/loc:@bidirectional_rnn/bw/bw/while/dropout/Cast*
_output_shapes
:*
	elem_type0
Ł
Dgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul/EnterEnterDgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul/f_acc*
parallel_iterations *
is_constant(*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:*
T0

Jgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul/StackPushV2StackPushV2Dgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul/Enter*bidirectional_rnn/bw/bw/while/dropout/Cast^gradients/Add_1*
_output_shapes
:	@*
swap_memory( *
T0
ě
Igradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul/StackPopV2
StackPopV2Ogradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul/StackPopV2/Enter^gradients/Sub_1*
	elem_type0*
_output_shapes
:	@
¸
Ogradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul/StackPopV2/EnterEnterDgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul/f_acc*
is_constant(*
T0*
_output_shapes
:*
parallel_iterations *E

frame_name75gradients/bidirectional_rnn/bw/bw/while/while_context

@gradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul_1MulNgradients/bidirectional_rnn/bw/bw/while/Select_grad/tuple/control_dependency_1Kgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul_1/StackPopV2*
T0*
_output_shapes
:	@
Ď
Fgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul_1/ConstConst*
_output_shapes
: *<
_class2
0.loc:@bidirectional_rnn/bw/bw/while/dropout/mul*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0

Fgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul_1/f_accStackV2Fgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul_1/Const*

stack_name *
_output_shapes
:*<
_class2
0.loc:@bidirectional_rnn/bw/bw/while/dropout/mul*
	elem_type0
§
Fgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul_1/EnterEnterFgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul_1/f_acc*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:*
parallel_iterations *
T0*
is_constant(

Lgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul_1/StackPushV2StackPushV2Fgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul_1/Enter)bidirectional_rnn/bw/bw/while/dropout/mul^gradients/Add_1*
_output_shapes
:	@*
swap_memory( *
T0
đ
Kgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul_1/StackPopV2
StackPopV2Qgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul_1/StackPopV2/Enter^gradients/Sub_1*
	elem_type0*
_output_shapes
:	@
ź
Qgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul_1/StackPopV2/EnterEnterFgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul_1/f_acc*
_output_shapes
:*E

frame_name75gradients/bidirectional_rnn/bw/bw/while/while_context*
T0*
parallel_iterations *
is_constant(
×
Kgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/tuple/group_depsNoOp?^gradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/MulA^gradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul_1
ę
Sgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/tuple/control_dependencyIdentity>gradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/MulL^gradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/tuple/group_deps*
_output_shapes
:	@*Q
_classG
ECloc:@gradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul*
T0
đ
Ugradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/tuple/control_dependency_1Identity@gradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul_1L^gradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/tuple/group_deps*S
_classI
GEloc:@gradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul_1*
T0*
_output_shapes
:	@
ä
Egradients/bidirectional_rnn/bw/bw/while/Switch_2_grad_1/NextIterationNextIterationkgradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1*
_output_shapes
: *
T0
Á
gradients/AddNAddNPgradients/bidirectional_rnn/fw/fw/while/Select_2_grad/tuple/control_dependency_1Qgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/tuple/control_dependency*Q
_classG
ECloc:@gradients/bidirectional_rnn/fw/fw/while/Select_2_grad/Select_1*
N*
_output_shapes
:	@*
T0
Ú
Fgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/MulMulgradients/AddNQgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2*
_output_shapes
:	@*
T0
ă
Lgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul/ConstConst*J
_class@
><loc:@bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid_2*
valueB :
˙˙˙˙˙˙˙˙˙*
_output_shapes
: *
dtype0
°
Lgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul/f_accStackV2Lgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul/Const*
_output_shapes
:*
	elem_type0*

stack_name *J
_class@
><loc:@bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid_2
ł
Lgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul/EnterEnterLgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context*
T0
ľ
Rgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul/StackPushV2StackPushV2Lgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul/Enter7bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid_2^gradients/Add*
_output_shapes
:	@*
swap_memory( *
T0
ú
Qgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2
StackPopV2Wgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:	@
Č
Wgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2/EnterEnterLgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*E

frame_name75gradients/bidirectional_rnn/fw/fw/while/while_context
Ţ
Hgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul_1Mulgradients/AddNSgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2*
_output_shapes
:	@*
T0
â
Ngradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul_1/ConstConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: *G
_class=
;9loc:@bidirectional_rnn/fw/fw/while/basic_lstm_cell/Tanh_1
ą
Ngradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul_1/f_accStackV2Ngradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul_1/Const*
	elem_type0*
_output_shapes
:*G
_class=
;9loc:@bidirectional_rnn/fw/fw/while/basic_lstm_cell/Tanh_1*

stack_name 
ˇ
Ngradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul_1/EnterEnterNgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul_1/f_acc*
parallel_iterations *;

frame_name-+bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:*
T0
ś
Tgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPushV2StackPushV2Ngradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul_1/Enter4bidirectional_rnn/fw/fw/while/basic_lstm_cell/Tanh_1^gradients/Add*
_output_shapes
:	@*
swap_memory( *
T0
ţ
Sgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2
StackPopV2Ygradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2/Enter^gradients/Sub*
_output_shapes
:	@*
	elem_type0
Ě
Ygradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2/EnterEnterNgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul_1/f_acc*E

frame_name75gradients/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
ď
Sgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/tuple/group_depsNoOpG^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/MulI^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul_1

[gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/tuple/control_dependencyIdentityFgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/MulT^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul*
_output_shapes
:	@

]gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/tuple/control_dependency_1IdentityHgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul_1T^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/tuple/group_deps*
T0*
_output_shapes
:	@*[
_classQ
OMloc:@gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul_1
Ą
>gradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/ShapeConst^gradients/Sub_1*
_output_shapes
:*
dtype0*
valueB"   @   

@gradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Shape_1Const^gradients/Sub_1*
valueB *
_output_shapes
: *
dtype0
Ś
Ngradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs>gradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Shape@gradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

<gradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/MulMulSgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/tuple/control_dependencyGgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/StackPopV2*
T0*
_output_shapes
:	@
Ď
Bgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/ConstConst*
dtype0*
_output_shapes
: *@
_class6
42loc:@bidirectional_rnn/bw/bw/while/dropout/truediv*
valueB :
˙˙˙˙˙˙˙˙˙

Bgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/f_accStackV2Bgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/Const*@
_class6
42loc:@bidirectional_rnn/bw/bw/while/dropout/truediv*

stack_name *
	elem_type0*
_output_shapes
:

Bgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/EnterEnterBgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/f_acc*
_output_shapes
:*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
T0*
parallel_iterations 

Hgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/StackPushV2StackPushV2Bgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/Enter-bidirectional_rnn/bw/bw/while/dropout/truediv^gradients/Add_1*
T0*
_output_shapes
: *
swap_memory( 
ß
Ggradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/StackPopV2
StackPopV2Mgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/StackPopV2/Enter^gradients/Sub_1*
_output_shapes
: *
	elem_type0
´
Mgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/StackPopV2/EnterEnterBgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/f_acc*
is_constant(*E

frame_name75gradients/bidirectional_rnn/bw/bw/while/while_context*
parallel_iterations *
_output_shapes
:*
T0

<gradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/SumSum<gradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/MulNgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs*
T0*
	keep_dims( *
_output_shapes
:	@*

Tidx0

@gradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/ReshapeReshape<gradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Sum>gradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Shape*
T0*
_output_shapes
:	@*
Tshape0

>gradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1MulIgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/StackPopV2Sgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/tuple/control_dependency*
T0*
_output_shapes
:	@
×
Dgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/ConstConst*F
_class<
:8loc:@bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2*
dtype0*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙

Dgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/f_accStackV2Dgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/Const*

stack_name *
	elem_type0*
_output_shapes
:*F
_class<
:8loc:@bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2
Ł
Dgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/EnterEnterDgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/f_acc*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context*
parallel_iterations *
T0*
_output_shapes
:*
is_constant(
Ł
Jgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/StackPushV2StackPushV2Dgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/Enter3bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2^gradients/Add_1*
_output_shapes
:	@*
swap_memory( *
T0
ě
Igradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/StackPopV2
StackPopV2Ogradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/StackPopV2/Enter^gradients/Sub_1*
_output_shapes
:	@*
	elem_type0
¸
Ogradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/StackPopV2/EnterEnterDgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/f_acc*
is_constant(*
_output_shapes
:*
parallel_iterations *
T0*E

frame_name75gradients/bidirectional_rnn/bw/bw/while/while_context

>gradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Sum_1Sum>gradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1Pgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *

Tidx0*
_output_shapes
: 
ţ
Bgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Reshape_1Reshape>gradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Sum_1@gradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
Ů
Igradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/tuple/group_depsNoOpA^gradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/ReshapeC^gradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Reshape_1
ę
Qgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/tuple/control_dependencyIdentity@gradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/ReshapeJ^gradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/tuple/group_deps*S
_classI
GEloc:@gradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Reshape*
T0*
_output_shapes
:	@
ç
Sgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/tuple/control_dependency_1IdentityBgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Reshape_1J^gradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Reshape_1*
_output_shapes
: 
´
Lgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Tanh_1_grad/TanhGradTanhGradSgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2[gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/tuple/control_dependency*
T0*
_output_shapes
:	@
˝
Rgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGradQgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2]gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/tuple/control_dependency_1*
_output_shapes
:	@*
T0
Ă
gradients/AddN_1AddNPgradients/bidirectional_rnn/bw/bw/while/Select_2_grad/tuple/control_dependency_1Qgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/tuple/control_dependency*
N*
_output_shapes
:	@*Q
_classG
ECloc:@gradients/bidirectional_rnn/bw/bw/while/Select_2_grad/Select_1*
T0
Ü
Fgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/MulMulgradients/AddN_1Qgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2*
T0*
_output_shapes
:	@
ă
Lgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul/ConstConst*
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙*
_output_shapes
: *J
_class@
><loc:@bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid_2
°
Lgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul/f_accStackV2Lgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul/Const*

stack_name *
	elem_type0*J
_class@
><loc:@bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid_2*
_output_shapes
:
ł
Lgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul/EnterEnterLgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul/f_acc*
T0*
is_constant(*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context*
parallel_iterations *
_output_shapes
:
ˇ
Rgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul/StackPushV2StackPushV2Lgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul/Enter7bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid_2^gradients/Add_1*
T0*
_output_shapes
:	@*
swap_memory( 
ü
Qgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2
StackPopV2Wgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2/Enter^gradients/Sub_1*
	elem_type0*
_output_shapes
:	@
Č
Wgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2/EnterEnterLgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul/f_acc*
_output_shapes
:*E

frame_name75gradients/bidirectional_rnn/bw/bw/while/while_context*
T0*
parallel_iterations *
is_constant(
ŕ
Hgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul_1Mulgradients/AddN_1Sgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2*
_output_shapes
:	@*
T0
â
Ngradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul_1/ConstConst*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*G
_class=
;9loc:@bidirectional_rnn/bw/bw/while/basic_lstm_cell/Tanh_1
ą
Ngradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul_1/f_accStackV2Ngradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul_1/Const*
	elem_type0*
_output_shapes
:*G
_class=
;9loc:@bidirectional_rnn/bw/bw/while/basic_lstm_cell/Tanh_1*

stack_name 
ˇ
Ngradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul_1/EnterEnterNgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul_1/f_acc*
T0*
_output_shapes
:*
parallel_iterations *;

frame_name-+bidirectional_rnn/bw/bw/while/while_context*
is_constant(
¸
Tgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPushV2StackPushV2Ngradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul_1/Enter4bidirectional_rnn/bw/bw/while/basic_lstm_cell/Tanh_1^gradients/Add_1*
swap_memory( *
_output_shapes
:	@*
T0

Sgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2
StackPopV2Ygradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2/Enter^gradients/Sub_1*
_output_shapes
:	@*
	elem_type0
Ě
Ygradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2/EnterEnterNgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul_1/f_acc*E

frame_name75gradients/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:*
parallel_iterations *
T0*
is_constant(
ď
Sgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/tuple/group_depsNoOpG^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/MulI^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul_1

[gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/tuple/control_dependencyIdentityFgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/MulT^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/tuple/group_deps*
T0*
_output_shapes
:	@*Y
_classO
MKloc:@gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul

]gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/tuple/control_dependency_1IdentityHgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul_1T^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/tuple/group_deps*
_output_shapes
:	@*[
_classQ
OMloc:@gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul_1*
T0
ž
gradients/AddN_2AddNPgradients/bidirectional_rnn/fw/fw/while/Select_1_grad/tuple/control_dependency_1Lgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Tanh_1_grad/TanhGrad*
T0*
N*Q
_classG
ECloc:@gradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select_1*
_output_shapes
:	@
n
Sgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_1_grad/tuple/group_depsNoOp^gradients/AddN_2
Ě
[gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_1_grad/tuple/control_dependencyIdentitygradients/AddN_2T^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_1_grad/tuple/group_deps*
T0*
_output_shapes
:	@*Q
_classG
ECloc:@gradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select_1
Î
]gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_1_grad/tuple/control_dependency_1Identitygradients/AddN_2T^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_1_grad/tuple/group_deps*
_output_shapes
:	@*Q
_classG
ECloc:@gradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select_1*
T0
´
Lgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Tanh_1_grad/TanhGradTanhGradSgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2[gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/tuple/control_dependency*
_output_shapes
:	@*
T0
˝
Rgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGradQgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2]gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/tuple/control_dependency_1*
T0*
_output_shapes
:	@
Ł
Dgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/MulMul[gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_1_grad/tuple/control_dependencyOgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2*
T0*
_output_shapes
:	@
ß
Jgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul/ConstConst*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*H
_class>
<:loc:@bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid
Ş
Jgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul/f_accStackV2Jgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul/Const*
_output_shapes
:*
	elem_type0*

stack_name *H
_class>
<:loc:@bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid
Ż
Jgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul/EnterEnterJgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul/f_acc*
T0*
parallel_iterations *;

frame_name-+bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:
Ż
Pgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul/StackPushV2StackPushV2Jgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul/Enter5bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid^gradients/Add*
swap_memory( *
_output_shapes
:	@*
T0
ö
Ogradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2
StackPopV2Ugradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:	@
Ä
Ugradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2/EnterEnterJgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul/f_acc*
_output_shapes
:*
parallel_iterations *
is_constant(*E

frame_name75gradients/bidirectional_rnn/fw/fw/while/while_context*
T0
§
Fgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul_1Mul[gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_1_grad/tuple/control_dependencyQgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2*
T0*
_output_shapes
:	@
Ô
Lgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul_1/ConstConst*
dtype0*;
_class1
/-loc:@bidirectional_rnn/fw/fw/while/Identity_3*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙
Ą
Lgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul_1/f_accStackV2Lgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul_1/Const*

stack_name *
_output_shapes
:*;
_class1
/-loc:@bidirectional_rnn/fw/fw/while/Identity_3*
	elem_type0
ł
Lgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul_1/EnterEnterLgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul_1/f_acc*
is_constant(*
_output_shapes
:*
parallel_iterations *;

frame_name-+bidirectional_rnn/fw/fw/while/while_context*
T0
Ś
Rgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul_1/StackPushV2StackPushV2Lgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul_1/Enter(bidirectional_rnn/fw/fw/while/Identity_3^gradients/Add*
swap_memory( *
T0*
_output_shapes
:	@
ú
Qgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2
StackPopV2Wgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:	@
Č
Wgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2/EnterEnterLgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul_1/f_acc*
parallel_iterations *
_output_shapes
:*
T0*E

frame_name75gradients/bidirectional_rnn/fw/fw/while/while_context*
is_constant(
é
Qgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/tuple/group_depsNoOpE^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/MulG^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul_1

Ygradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/tuple/control_dependencyIdentityDgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/MulR^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/tuple/group_deps*
_output_shapes
:	@*W
_classM
KIloc:@gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul*
T0

[gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/tuple/control_dependency_1IdentityFgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul_1R^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/tuple/group_deps*Y
_classO
MKloc:@gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul_1*
T0*
_output_shapes
:	@
Š
Fgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/MulMul]gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_1_grad/tuple/control_dependency_1Qgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2*
T0*
_output_shapes
:	@
Ţ
Lgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul/ConstConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*E
_class;
97loc:@bidirectional_rnn/fw/fw/while/basic_lstm_cell/Tanh*
_output_shapes
: 
Ť
Lgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul/f_accStackV2Lgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul/Const*
	elem_type0*E
_class;
97loc:@bidirectional_rnn/fw/fw/while/basic_lstm_cell/Tanh*

stack_name *
_output_shapes
:
ł
Lgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul/EnterEnterLgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul/f_acc*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:*
T0*
parallel_iterations *
is_constant(
°
Rgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul/StackPushV2StackPushV2Lgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul/Enter2bidirectional_rnn/fw/fw/while/basic_lstm_cell/Tanh^gradients/Add*
T0*
_output_shapes
:	@*
swap_memory( 
ú
Qgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2
StackPopV2Wgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2/Enter^gradients/Sub*
_output_shapes
:	@*
	elem_type0
Č
Wgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2/EnterEnterLgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul/f_acc*
_output_shapes
:*
T0*
parallel_iterations *
is_constant(*E

frame_name75gradients/bidirectional_rnn/fw/fw/while/while_context
­
Hgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul_1Mul]gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_1_grad/tuple/control_dependency_1Sgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2*
T0*
_output_shapes
:	@
ĺ
Ngradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul_1/ConstConst*
_output_shapes
: *
dtype0*J
_class@
><loc:@bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid_1*
valueB :
˙˙˙˙˙˙˙˙˙
´
Ngradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul_1/f_accStackV2Ngradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul_1/Const*
	elem_type0*
_output_shapes
:*J
_class@
><loc:@bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid_1*

stack_name 
ˇ
Ngradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul_1/EnterEnterNgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul_1/f_acc*
parallel_iterations *;

frame_name-+bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:*
is_constant(*
T0
š
Tgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPushV2StackPushV2Ngradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul_1/Enter7bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid_1^gradients/Add*
swap_memory( *
T0*
_output_shapes
:	@
ţ
Sgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2
StackPopV2Ygradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:	@
Ě
Ygradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2/EnterEnterNgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul_1/f_acc*
parallel_iterations *
T0*
_output_shapes
:*E

frame_name75gradients/bidirectional_rnn/fw/fw/while/while_context*
is_constant(
ď
Sgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/tuple/group_depsNoOpG^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/MulI^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul_1

[gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/tuple/control_dependencyIdentityFgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/MulT^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/tuple/group_deps*
T0*
_output_shapes
:	@*Y
_classO
MKloc:@gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul

]gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/tuple/control_dependency_1IdentityHgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul_1T^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/tuple/group_deps*[
_classQ
OMloc:@gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul_1*
T0*
_output_shapes
:	@
ž
gradients/AddN_3AddNPgradients/bidirectional_rnn/bw/bw/while/Select_1_grad/tuple/control_dependency_1Lgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Tanh_1_grad/TanhGrad*
T0*
_output_shapes
:	@*
N*Q
_classG
ECloc:@gradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select_1
n
Sgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_1_grad/tuple/group_depsNoOp^gradients/AddN_3
Ě
[gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_1_grad/tuple/control_dependencyIdentitygradients/AddN_3T^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_1_grad/tuple/group_deps*
T0*
_output_shapes
:	@*Q
_classG
ECloc:@gradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select_1
Î
]gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_1_grad/tuple/control_dependency_1Identitygradients/AddN_3T^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_1_grad/tuple/group_deps*Q
_classG
ECloc:@gradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select_1*
T0*
_output_shapes
:	@
Ç
gradients/AddN_4AddNNgradients/bidirectional_rnn/fw/fw/while/Select_1_grad/tuple/control_dependencyYgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/tuple/control_dependency*
T0*
_output_shapes
:	@*O
_classE
CAloc:@gradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select*
N
ˇ
Pgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGradOgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2[gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/tuple/control_dependency_1*
_output_shapes
:	@*
T0
˝
Rgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGradSgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2[gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/tuple/control_dependency*
T0*
_output_shapes
:	@
˛
Jgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Tanh_grad/TanhGradTanhGradQgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2]gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/tuple/control_dependency_1*
T0*
_output_shapes
:	@
Ł
Dgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/MulMul[gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_1_grad/tuple/control_dependencyOgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2*
_output_shapes
:	@*
T0
ß
Jgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul/ConstConst*
dtype0*
_output_shapes
: *H
_class>
<:loc:@bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid*
valueB :
˙˙˙˙˙˙˙˙˙
Ş
Jgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul/f_accStackV2Jgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul/Const*

stack_name *
	elem_type0*H
_class>
<:loc:@bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid*
_output_shapes
:
Ż
Jgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul/EnterEnterJgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul/f_acc*
parallel_iterations *
is_constant(*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context*
T0*
_output_shapes
:
ą
Pgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul/StackPushV2StackPushV2Jgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul/Enter5bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid^gradients/Add_1*
_output_shapes
:	@*
T0*
swap_memory( 
ř
Ogradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2
StackPopV2Ugradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2/Enter^gradients/Sub_1*
_output_shapes
:	@*
	elem_type0
Ä
Ugradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2/EnterEnterJgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul/f_acc*
_output_shapes
:*
is_constant(*
T0*E

frame_name75gradients/bidirectional_rnn/bw/bw/while/while_context*
parallel_iterations 
§
Fgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul_1Mul[gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_1_grad/tuple/control_dependencyQgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2*
_output_shapes
:	@*
T0
Ô
Lgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul_1/ConstConst*
valueB :
˙˙˙˙˙˙˙˙˙*
_output_shapes
: *
dtype0*;
_class1
/-loc:@bidirectional_rnn/bw/bw/while/Identity_3
Ą
Lgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul_1/f_accStackV2Lgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul_1/Const*
_output_shapes
:*
	elem_type0*;
_class1
/-loc:@bidirectional_rnn/bw/bw/while/Identity_3*

stack_name 
ł
Lgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul_1/EnterEnterLgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul_1/f_acc*
_output_shapes
:*
T0*
is_constant(*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context*
parallel_iterations 
¨
Rgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul_1/StackPushV2StackPushV2Lgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul_1/Enter(bidirectional_rnn/bw/bw/while/Identity_3^gradients/Add_1*
_output_shapes
:	@*
swap_memory( *
T0
ü
Qgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2
StackPopV2Wgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2/Enter^gradients/Sub_1*
_output_shapes
:	@*
	elem_type0
Č
Wgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2/EnterEnterLgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul_1/f_acc*
T0*
parallel_iterations *E

frame_name75gradients/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:
é
Qgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/tuple/group_depsNoOpE^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/MulG^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul_1

Ygradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/tuple/control_dependencyIdentityDgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/MulR^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/tuple/group_deps*
_output_shapes
:	@*W
_classM
KIloc:@gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul*
T0

[gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/tuple/control_dependency_1IdentityFgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul_1R^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/tuple/group_deps*Y
_classO
MKloc:@gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul_1*
_output_shapes
:	@*
T0
Š
Fgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/MulMul]gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_1_grad/tuple/control_dependency_1Qgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2*
_output_shapes
:	@*
T0
Ţ
Lgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul/ConstConst*
valueB :
˙˙˙˙˙˙˙˙˙*E
_class;
97loc:@bidirectional_rnn/bw/bw/while/basic_lstm_cell/Tanh*
dtype0*
_output_shapes
: 
Ť
Lgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul/f_accStackV2Lgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul/Const*E
_class;
97loc:@bidirectional_rnn/bw/bw/while/basic_lstm_cell/Tanh*
_output_shapes
:*
	elem_type0*

stack_name 
ł
Lgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul/EnterEnterLgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul/f_acc*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context*
T0*
parallel_iterations *
_output_shapes
:*
is_constant(
˛
Rgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul/StackPushV2StackPushV2Lgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul/Enter2bidirectional_rnn/bw/bw/while/basic_lstm_cell/Tanh^gradients/Add_1*
swap_memory( *
_output_shapes
:	@*
T0
ü
Qgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2
StackPopV2Wgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2/Enter^gradients/Sub_1*
_output_shapes
:	@*
	elem_type0
Č
Wgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2/EnterEnterLgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul/f_acc*
is_constant(*
_output_shapes
:*
T0*E

frame_name75gradients/bidirectional_rnn/bw/bw/while/while_context*
parallel_iterations 
­
Hgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul_1Mul]gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_1_grad/tuple/control_dependency_1Sgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2*
_output_shapes
:	@*
T0
ĺ
Ngradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul_1/ConstConst*J
_class@
><loc:@bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid_1*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
´
Ngradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul_1/f_accStackV2Ngradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul_1/Const*J
_class@
><loc:@bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid_1*
	elem_type0*

stack_name *
_output_shapes
:
ˇ
Ngradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul_1/EnterEnterNgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul_1/f_acc*
parallel_iterations *
is_constant(*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:*
T0
ť
Tgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPushV2StackPushV2Ngradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul_1/Enter7bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid_1^gradients/Add_1*
_output_shapes
:	@*
swap_memory( *
T0

Sgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2
StackPopV2Ygradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2/Enter^gradients/Sub_1*
	elem_type0*
_output_shapes
:	@
Ě
Ygradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2/EnterEnterNgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul_1/f_acc*
is_constant(*
T0*
_output_shapes
:*
parallel_iterations *E

frame_name75gradients/bidirectional_rnn/bw/bw/while/while_context
ď
Sgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/tuple/group_depsNoOpG^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/MulI^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul_1

[gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/tuple/control_dependencyIdentityFgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/MulT^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/tuple/group_deps*Y
_classO
MKloc:@gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul*
T0*
_output_shapes
:	@

]gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/tuple/control_dependency_1IdentityHgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul_1T^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/tuple/group_deps*[
_classQ
OMloc:@gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul_1*
_output_shapes
:	@*
T0

Egradients/bidirectional_rnn/fw/fw/while/Switch_3_grad_1/NextIterationNextIterationgradients/AddN_4*
T0*
_output_shapes
:	@
§
Fgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
dtype0*
valueB"   @   

Hgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_grad/Shape_1Const^gradients/Sub*
dtype0*
_output_shapes
: *
valueB 
ž
Vgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_grad/BroadcastGradientArgsBroadcastGradientArgsFgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_grad/ShapeHgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ź
Dgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_grad/SumSumPgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid_grad/SigmoidGradVgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs*
T0*
	keep_dims( *
_output_shapes
:	@*

Tidx0

Hgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_grad/ReshapeReshapeDgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_grad/SumFgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_grad/Shape*
_output_shapes
:	@*
Tshape0*
T0
ˇ
Fgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_grad/Sum_1SumPgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid_grad/SigmoidGradXgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *

Tidx0*
_output_shapes
: 

Jgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_grad/Reshape_1ReshapeFgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_grad/Sum_1Hgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
ń
Qgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_grad/tuple/group_depsNoOpI^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_grad/ReshapeK^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_grad/Reshape_1

Ygradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_grad/tuple/control_dependencyIdentityHgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_grad/ReshapeR^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_grad/tuple/group_deps*
_output_shapes
:	@*
T0*[
_classQ
OMloc:@gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_grad/Reshape

[gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_grad/tuple/control_dependency_1IdentityJgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_grad/Reshape_1R^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_grad/Reshape_1*
_output_shapes
: 
Ç
gradients/AddN_5AddNNgradients/bidirectional_rnn/bw/bw/while/Select_1_grad/tuple/control_dependencyYgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/tuple/control_dependency*
T0*
N*
_output_shapes
:	@*O
_classE
CAloc:@gradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select
ˇ
Pgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGradOgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2[gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/tuple/control_dependency_1*
_output_shapes
:	@*
T0
˝
Rgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGradSgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2[gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/tuple/control_dependency*
T0*
_output_shapes
:	@
˛
Jgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Tanh_grad/TanhGradTanhGradQgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2]gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/tuple/control_dependency_1*
T0*
_output_shapes
:	@
ľ
Igradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/split_grad/concatConcatV2Rgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid_1_grad/SigmoidGradJgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Tanh_grad/TanhGradYgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_grad/tuple/control_dependencyRgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid_2_grad/SigmoidGradOgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/split_grad/concat/Const*
T0*

Tidx0* 
_output_shapes
:
*
N
Ą
Ogradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/split_grad/concat/ConstConst^gradients/Sub*
dtype0*
_output_shapes
: *
value	B :

Egradients/bidirectional_rnn/bw/bw/while/Switch_3_grad_1/NextIterationNextIterationgradients/AddN_5*
T0*
_output_shapes
:	@
Š
Fgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_grad/ShapeConst^gradients/Sub_1*
valueB"   @   *
dtype0*
_output_shapes
:

Hgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_grad/Shape_1Const^gradients/Sub_1*
dtype0*
_output_shapes
: *
valueB 
ž
Vgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_grad/BroadcastGradientArgsBroadcastGradientArgsFgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_grad/ShapeHgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ź
Dgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_grad/SumSumPgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid_grad/SigmoidGradVgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs*

Tidx0*
T0*
_output_shapes
:	@*
	keep_dims( 

Hgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_grad/ReshapeReshapeDgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_grad/SumFgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_grad/Shape*
_output_shapes
:	@*
T0*
Tshape0
ˇ
Fgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_grad/Sum_1SumPgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid_grad/SigmoidGradXgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
: *
	keep_dims( *
T0

Jgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_grad/Reshape_1ReshapeFgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_grad/Sum_1Hgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
ń
Qgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_grad/tuple/group_depsNoOpI^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_grad/ReshapeK^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_grad/Reshape_1

Ygradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_grad/tuple/control_dependencyIdentityHgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_grad/ReshapeR^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_grad/tuple/group_deps*[
_classQ
OMloc:@gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_grad/Reshape*
_output_shapes
:	@*
T0

[gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_grad/tuple/control_dependency_1IdentityJgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_grad/Reshape_1R^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_grad/tuple/group_deps*]
_classS
QOloc:@gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_grad/Reshape_1*
_output_shapes
: *
T0
ç
Pgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGradIgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/split_grad/concat*
data_formatNHWC*
T0*
_output_shapes	
:
ü
Ugradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd_grad/tuple/group_depsNoOpQ^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd_grad/BiasAddGradJ^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/split_grad/concat

]gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependencyIdentityIgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/split_grad/concatV^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd_grad/tuple/group_deps*\
_classR
PNloc:@gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/split_grad/concat*
T0* 
_output_shapes
:

 
_gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependency_1IdentityPgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd_grad/BiasAddGradV^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
ľ
Igradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/split_grad/concatConcatV2Rgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid_1_grad/SigmoidGradJgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Tanh_grad/TanhGradYgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_grad/tuple/control_dependencyRgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid_2_grad/SigmoidGradOgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/split_grad/concat/Const*

Tidx0* 
_output_shapes
:
*
T0*
N
Ł
Ogradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/split_grad/concat/ConstConst^gradients/Sub_1*
dtype0*
value	B :*
_output_shapes
: 
Ö
Jgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/MatMulMatMul]gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependencyPgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/MatMul/Enter* 
_output_shapes
:
Ŕ*
T0*
transpose_a( *
transpose_b(
Ť
Pgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/MatMul/EnterEnter0bidirectional_rnn/fw/basic_lstm_cell/kernel/read* 
_output_shapes
:
Ŕ*
is_constant(*
parallel_iterations *E

frame_name75gradients/bidirectional_rnn/fw/fw/while/while_context*
T0
ß
Lgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/MatMul_1MatMulWgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2]gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
Ŕ*
T0*
transpose_b( *
transpose_a(
ć
Rgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/MatMul_1/ConstConst*
valueB :
˙˙˙˙˙˙˙˙˙*G
_class=
;9loc:@bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat*
_output_shapes
: *
dtype0
š
Rgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/MatMul_1/f_accStackV2Rgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/MatMul_1/Const*G
_class=
;9loc:@bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat*

stack_name *
_output_shapes
:*
	elem_type0
ż
Rgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/MatMul_1/EnterEnterRgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc*
parallel_iterations *
_output_shapes
:*
is_constant(*
T0*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context
ż
Xgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2StackPushV2Rgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/MatMul_1/Enter4bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat^gradients/Add*
swap_memory( *
T0* 
_output_shapes
:
Ŕ

Wgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2
StackPopV2]gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2/Enter^gradients/Sub* 
_output_shapes
:
Ŕ*
	elem_type0
Ô
]gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2/EnterEnterRgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc*
parallel_iterations *E

frame_name75gradients/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:*
T0
ř
Tgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/tuple/group_depsNoOpK^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/MatMulM^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/MatMul_1

\gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/tuple/control_dependencyIdentityJgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/MatMulU^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/MatMul* 
_output_shapes
:
Ŕ

^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/tuple/control_dependency_1IdentityLgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/MatMul_1U^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/tuple/group_deps* 
_output_shapes
:
Ŕ*_
_classU
SQloc:@gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/MatMul_1*
T0

Pgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd/Enter_grad/b_accConst*
dtype0*
valueB*    *
_output_shapes	
:
Č
Rgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1EnterPgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc*
parallel_iterations *E

frame_name75gradients/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes	
:*
is_constant( *
T0
ş
Rgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2MergeRgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1Xgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd/Enter_grad/NextIteration*
_output_shapes
	:: *
N*
T0
ń
Qgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd/Enter_grad/SwitchSwitchRgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2gradients/b_count_2*"
_output_shapes
::*
T0
ą
Ngradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd/Enter_grad/AddAddSgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd/Enter_grad/Switch:1_gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes	
:*
T0
ß
Xgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd/Enter_grad/NextIterationNextIterationNgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd/Enter_grad/Add*
_output_shapes	
:*
T0
Ó
Rgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3ExitQgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd/Enter_grad/Switch*
T0*
_output_shapes	
:
ç
Pgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGradIgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/split_grad/concat*
T0*
data_formatNHWC*
_output_shapes	
:
ü
Ugradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd_grad/tuple/group_depsNoOpQ^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd_grad/BiasAddGradJ^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/split_grad/concat

]gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependencyIdentityIgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/split_grad/concatV^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd_grad/tuple/group_deps*\
_classR
PNloc:@gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/split_grad/concat* 
_output_shapes
:
*
T0
 
_gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependency_1IdentityPgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd_grad/BiasAddGradV^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*c
_classY
WUloc:@gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd_grad/BiasAddGrad*
T0

Igradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/ConstConst^gradients/Sub*
_output_shapes
: *
dtype0*
value	B :

Hgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/RankConst^gradients/Sub*
value	B :*
_output_shapes
: *
dtype0

Ggradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/modFloorModIgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/ConstHgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/Rank*
_output_shapes
: *
T0
Ş
Igradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
dtype0*
valueB"      
Ź
Kgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/Shape_1Const^gradients/Sub*
valueB"   @   *
_output_shapes
:*
dtype0
ě
Pgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/ConcatOffsetConcatOffsetGgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/modIgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/ShapeKgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/Shape_1*
N* 
_output_shapes
::

Igradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/SliceSlice\gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/tuple/control_dependencyPgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/ConcatOffsetIgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/Shape*
T0* 
_output_shapes
:
*
Index0

Kgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/Slice_1Slice\gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/tuple/control_dependencyRgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/ConcatOffset:1Kgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/Shape_1*
_output_shapes
:	@*
T0*
Index0
ö
Tgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/tuple/group_depsNoOpJ^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/SliceL^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/Slice_1

\gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/tuple/control_dependencyIdentityIgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/SliceU^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/tuple/group_deps*\
_classR
PNloc:@gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/Slice* 
_output_shapes
:
*
T0

^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/tuple/control_dependency_1IdentityKgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/Slice_1U^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/tuple/group_deps*
T0*
_output_shapes
:	@*^
_classT
RPloc:@gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/Slice_1
¨
Ogradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul/Enter_grad/b_accConst*
valueB
Ŕ*    * 
_output_shapes
:
Ŕ*
dtype0
Ë
Qgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_1EnterOgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul/Enter_grad/b_acc*
parallel_iterations *
T0* 
_output_shapes
:
Ŕ*
is_constant( *E

frame_name75gradients/bidirectional_rnn/fw/fw/while/while_context
ź
Qgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_2MergeQgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_1Wgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul/Enter_grad/NextIteration*"
_output_shapes
:
Ŕ: *
T0*
N
ů
Pgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul/Enter_grad/SwitchSwitchQgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_2gradients/b_count_2*
T0*,
_output_shapes
:
Ŕ:
Ŕ
ł
Mgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul/Enter_grad/AddAddRgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul/Enter_grad/Switch:1^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/tuple/control_dependency_1* 
_output_shapes
:
Ŕ*
T0
â
Wgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul/Enter_grad/NextIterationNextIterationMgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul/Enter_grad/Add* 
_output_shapes
:
Ŕ*
T0
Ö
Qgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_3ExitPgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul/Enter_grad/Switch* 
_output_shapes
:
Ŕ*
T0
Ö
Jgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/MatMulMatMul]gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependencyPgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/MatMul/Enter*
transpose_a( *
T0*
transpose_b(* 
_output_shapes
:
Ŕ
Ť
Pgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/MatMul/EnterEnter0bidirectional_rnn/bw/basic_lstm_cell/kernel/read*
T0*E

frame_name75gradients/bidirectional_rnn/bw/bw/while/while_context*
is_constant(* 
_output_shapes
:
Ŕ*
parallel_iterations 
ß
Lgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/MatMul_1MatMulWgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2]gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
T0* 
_output_shapes
:
Ŕ*
transpose_b( 
ć
Rgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/MatMul_1/ConstConst*
dtype0*
_output_shapes
: *G
_class=
;9loc:@bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat*
valueB :
˙˙˙˙˙˙˙˙˙
š
Rgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/MatMul_1/f_accStackV2Rgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/MatMul_1/Const*

stack_name *
_output_shapes
:*
	elem_type0*G
_class=
;9loc:@bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat
ż
Rgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/MatMul_1/EnterEnterRgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc*
parallel_iterations *
T0*
_output_shapes
:*
is_constant(*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context
Á
Xgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2StackPushV2Rgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/MatMul_1/Enter4bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat^gradients/Add_1*
T0* 
_output_shapes
:
Ŕ*
swap_memory( 

Wgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2
StackPopV2]gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2/Enter^gradients/Sub_1*
	elem_type0* 
_output_shapes
:
Ŕ
Ô
]gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2/EnterEnterRgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*E

frame_name75gradients/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:*
is_constant(*
parallel_iterations 
ř
Tgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/tuple/group_depsNoOpK^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/MatMulM^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/MatMul_1

\gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/tuple/control_dependencyIdentityJgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/MatMulU^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/tuple/group_deps*
T0* 
_output_shapes
:
Ŕ*]
_classS
QOloc:@gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/MatMul

^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/tuple/control_dependency_1IdentityLgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/MatMul_1U^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/tuple/group_deps* 
_output_shapes
:
Ŕ*_
_classU
SQloc:@gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/MatMul_1*
T0

Pgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd/Enter_grad/b_accConst*
_output_shapes	
:*
valueB*    *
dtype0
Č
Rgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1EnterPgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc*
is_constant( *
parallel_iterations *
T0*
_output_shapes	
:*E

frame_name75gradients/bidirectional_rnn/bw/bw/while/while_context
ş
Rgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2MergeRgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1Xgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd/Enter_grad/NextIteration*
T0*
N*
_output_shapes
	:: 
ń
Qgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd/Enter_grad/SwitchSwitchRgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2gradients/b_count_6*"
_output_shapes
::*
T0
ą
Ngradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd/Enter_grad/AddAddSgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd/Enter_grad/Switch:1_gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:
ß
Xgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd/Enter_grad/NextIterationNextIterationNgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd/Enter_grad/Add*
T0*
_output_shapes	
:
Ó
Rgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3ExitQgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd/Enter_grad/Switch*
T0*
_output_shapes	
:
Ö
`gradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3fgradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enterhgradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1^gradients/Sub*H
_class>
<:loc:@bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter*
_output_shapes

:: *
source	gradients
ú
fgradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnter%bidirectional_rnn/fw/fw/TensorArray_1*E

frame_name75gradients/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*H
_class>
<:loc:@bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter*
_output_shapes
:*
parallel_iterations *
T0
Ľ
hgradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1EnterRbidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*H
_class>
<:loc:@bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter*
parallel_iterations *
_output_shapes
: *E

frame_name75gradients/bidirectional_rnn/fw/fw/while/while_context*
T0*
is_constant(
 
\gradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flowIdentityhgradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1a^gradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3*H
_class>
<:loc:@bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter*
T0*
_output_shapes
: 
Ś
bgradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3`gradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3mgradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2\gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/tuple/control_dependency\gradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flow*
T0*
_output_shapes
: 
Ě
gradients/AddN_6AddNNgradients/bidirectional_rnn/fw/fw/while/Select_2_grad/tuple/control_dependency^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/tuple/control_dependency_1*
T0*O
_classE
CAloc:@gradients/bidirectional_rnn/fw/fw/while/Select_2_grad/Select*
N*
_output_shapes
:	@

Igradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/ConstConst^gradients/Sub_1*
_output_shapes
: *
value	B :*
dtype0

Hgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/RankConst^gradients/Sub_1*
value	B :*
_output_shapes
: *
dtype0

Ggradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/modFloorModIgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/ConstHgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/Rank*
T0*
_output_shapes
: 
Ź
Igradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/ShapeConst^gradients/Sub_1*
valueB"      *
_output_shapes
:*
dtype0
Ž
Kgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/Shape_1Const^gradients/Sub_1*
valueB"   @   *
dtype0*
_output_shapes
:
ě
Pgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/ConcatOffsetConcatOffsetGgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/modIgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/ShapeKgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/Shape_1*
N* 
_output_shapes
::

Igradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/SliceSlice\gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/tuple/control_dependencyPgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/ConcatOffsetIgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/Shape*
Index0*
T0* 
_output_shapes
:


Kgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/Slice_1Slice\gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/tuple/control_dependencyRgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/ConcatOffset:1Kgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/Shape_1*
Index0*
T0*
_output_shapes
:	@
ö
Tgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/tuple/group_depsNoOpJ^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/SliceL^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/Slice_1

\gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/tuple/control_dependencyIdentityIgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/SliceU^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/tuple/group_deps*\
_classR
PNloc:@gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/Slice* 
_output_shapes
:
*
T0

^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/tuple/control_dependency_1IdentityKgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/Slice_1U^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/tuple/group_deps*^
_classT
RPloc:@gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/Slice_1*
_output_shapes
:	@*
T0
¨
Ogradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul/Enter_grad/b_accConst*
valueB
Ŕ*    *
dtype0* 
_output_shapes
:
Ŕ
Ë
Qgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_1EnterOgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul/Enter_grad/b_acc*E

frame_name75gradients/bidirectional_rnn/bw/bw/while/while_context*
is_constant( * 
_output_shapes
:
Ŕ*
parallel_iterations *
T0
ź
Qgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_2MergeQgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_1Wgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul/Enter_grad/NextIteration*
T0*
N*"
_output_shapes
:
Ŕ: 
ů
Pgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul/Enter_grad/SwitchSwitchQgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_2gradients/b_count_6*,
_output_shapes
:
Ŕ:
Ŕ*
T0
ł
Mgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul/Enter_grad/AddAddRgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul/Enter_grad/Switch:1^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:
Ŕ
â
Wgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul/Enter_grad/NextIterationNextIterationMgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul/Enter_grad/Add* 
_output_shapes
:
Ŕ*
T0
Ö
Qgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_3ExitPgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul/Enter_grad/Switch* 
_output_shapes
:
Ŕ*
T0

Lgradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_accConst*
dtype0*
valueB
 *    *
_output_shapes
: 
ť
Ngradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_acc_1EnterLgradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_acc*
T0*E

frame_name75gradients/bidirectional_rnn/fw/fw/while/while_context*
parallel_iterations *
_output_shapes
: *
is_constant( 
Š
Ngradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_acc_2MergeNgradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Tgradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/NextIteration*
_output_shapes
: : *
T0*
N
ß
Mgradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/SwitchSwitchNgradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_acc_2gradients/b_count_2*
_output_shapes
: : *
T0
§
Jgradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/AddAddOgradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/Switch:1bgradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3*
_output_shapes
: *
T0
Ň
Tgradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/NextIterationNextIterationJgradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/Add*
T0*
_output_shapes
: 
Ć
Ngradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_acc_3ExitMgradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/Switch*
T0*
_output_shapes
: 

Egradients/bidirectional_rnn/fw/fw/while/Switch_4_grad_1/NextIterationNextIterationgradients/AddN_6*
_output_shapes
:	@*
T0
Ř
`gradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3fgradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enterhgradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1^gradients/Sub_1*
source	gradients*H
_class>
<:loc:@bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter*
_output_shapes

:: 
ú
fgradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnter%bidirectional_rnn/bw/bw/TensorArray_1*H
_class>
<:loc:@bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter*
is_constant(*
_output_shapes
:*
parallel_iterations *
T0*E

frame_name75gradients/bidirectional_rnn/bw/bw/while/while_context
Ľ
hgradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1EnterRbidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*H
_class>
<:loc:@bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter*
is_constant(*
T0*E

frame_name75gradients/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
: *
parallel_iterations 
 
\gradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flowIdentityhgradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1a^gradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*H
_class>
<:loc:@bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter*
_output_shapes
: 
Ś
bgradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3`gradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3mgradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2\gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/tuple/control_dependency\gradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flow*
_output_shapes
: *
T0
Ě
gradients/AddN_7AddNNgradients/bidirectional_rnn/bw/bw/while/Select_2_grad/tuple/control_dependency^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/tuple/control_dependency_1*O
_classE
CAloc:@gradients/bidirectional_rnn/bw/bw/while/Select_2_grad/Select*
N*
_output_shapes
:	@*
T0
˙
gradients/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3%bidirectional_rnn/fw/fw/TensorArray_1Ngradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*
_output_shapes

:: *8
_class.
,*loc:@bidirectional_rnn/fw/fw/TensorArray_1*
source	gradients
˝
gradients/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flowIdentityNgradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_acc_3^gradients/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*
_output_shapes
: *8
_class.
,*loc:@bidirectional_rnn/fw/fw/TensorArray_1

ugradients/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3TensorArrayGatherV3gradients/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV30bidirectional_rnn/fw/fw/TensorArrayUnstack/rangegradients/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flow*
element_shape:*
dtype0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ă
rgradients/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_depsNoOpv^gradients/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3O^gradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_acc_3
ľ
zgradients/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependencyIdentityugradients/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3s^gradients/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*
_class~
|zloc:@gradients/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ń
|gradients/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependency_1IdentityNgradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_acc_3s^gradients/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*
_output_shapes
: *
T0*a
_classW
USloc:@gradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_acc_3

Lgradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_accConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ť
Ngradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_acc_1EnterLgradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_acc*
is_constant( *
T0*
_output_shapes
: *
parallel_iterations *E

frame_name75gradients/bidirectional_rnn/bw/bw/while/while_context
Š
Ngradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_acc_2MergeNgradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Tgradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/NextIteration*
T0*
_output_shapes
: : *
N
ß
Mgradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/SwitchSwitchNgradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_acc_2gradients/b_count_6*
T0*
_output_shapes
: : 
§
Jgradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/AddAddOgradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/Switch:1bgradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3*
_output_shapes
: *
T0
Ň
Tgradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/NextIterationNextIterationJgradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/Add*
T0*
_output_shapes
: 
Ć
Ngradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_acc_3ExitMgradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/Switch*
T0*
_output_shapes
: 

Egradients/bidirectional_rnn/bw/bw/while/Switch_4_grad_1/NextIterationNextIterationgradients/AddN_7*
T0*
_output_shapes
:	@

Bgradients/bidirectional_rnn/fw/fw/transpose_grad/InvertPermutationInvertPermutationbidirectional_rnn/fw/fw/concat*
T0*
_output_shapes
:
Ě
:gradients/bidirectional_rnn/fw/fw/transpose_grad/transpose	Transposezgradients/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependencyBgradients/bidirectional_rnn/fw/fw/transpose_grad/InvertPermutation*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tperm0
˙
gradients/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3%bidirectional_rnn/bw/bw/TensorArray_1Ngradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*8
_class.
,*loc:@bidirectional_rnn/bw/bw/TensorArray_1*
source	gradients*
_output_shapes

:: 
˝
gradients/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flowIdentityNgradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_acc_3^gradients/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*8
_class.
,*loc:@bidirectional_rnn/bw/bw/TensorArray_1*
_output_shapes
: 

ugradients/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3TensorArrayGatherV3gradients/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV30bidirectional_rnn/bw/bw/TensorArrayUnstack/rangegradients/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flow*
dtype0*
element_shape:*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ă
rgradients/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_depsNoOpv^gradients/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3O^gradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_acc_3
ľ
zgradients/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependencyIdentityugradients/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3s^gradients/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*
_class~
|zloc:@gradients/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ń
|gradients/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependency_1IdentityNgradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_acc_3s^gradients/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*
_output_shapes
: *a
_classW
USloc:@gradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*
T0

Bgradients/bidirectional_rnn/bw/bw/transpose_grad/InvertPermutationInvertPermutationbidirectional_rnn/bw/bw/concat*
T0*
_output_shapes
:
Ě
:gradients/bidirectional_rnn/bw/bw/transpose_grad/transpose	Transposezgradients/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependencyBgradients/bidirectional_rnn/bw/bw/transpose_grad/InvertPermutation*-
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tperm0*
T0

Cgradients/bidirectional_rnn/bw/ReverseSequence_grad/ReverseSequenceReverseSequence:gradients/bidirectional_rnn/bw/bw/transpose_grad/transposePlaceholder_2*
	batch_dim *

Tlen0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
seq_dim
Š
gradients/AddN_8AddN:gradients/bidirectional_rnn/fw/fw/transpose_grad/transposeCgradients/bidirectional_rnn/bw/ReverseSequence_grad/ReverseSequence*-
_output_shapes
:˙˙˙˙˙˙˙˙˙*M
_classC
A?loc:@gradients/bidirectional_rnn/fw/fw/transpose_grad/transpose*
N*
T0

%gradients/embedding_lookup_grad/ShapeConst*
dtype0	*
_class
loc:@Embedding*
_output_shapes
:*%
valueB	"              
ľ
$gradients/embedding_lookup_grad/CastCast%gradients/embedding_lookup_grad/Shape*

DstT0*
_output_shapes
:*
_class
loc:@Embedding*
Truncate( *

SrcT0	
j
$gradients/embedding_lookup_grad/SizeSizePlaceholder*
_output_shapes
: *
out_type0*
T0
p
.gradients/embedding_lookup_grad/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
value	B : 
ż
*gradients/embedding_lookup_grad/ExpandDims
ExpandDims$gradients/embedding_lookup_grad/Size.gradients/embedding_lookup_grad/ExpandDims/dim*
_output_shapes
:*
T0*

Tdim0
}
3gradients/embedding_lookup_grad/strided_slice/stackConst*
dtype0*
valueB:*
_output_shapes
:

5gradients/embedding_lookup_grad/strided_slice/stack_1Const*
_output_shapes
:*
valueB: *
dtype0

5gradients/embedding_lookup_grad/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0

-gradients/embedding_lookup_grad/strided_sliceStridedSlice$gradients/embedding_lookup_grad/Cast3gradients/embedding_lookup_grad/strided_slice/stack5gradients/embedding_lookup_grad/strided_slice/stack_15gradients/embedding_lookup_grad/strided_slice/stack_2*
T0*
shrink_axis_mask *
Index0*
_output_shapes
:*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask
m
+gradients/embedding_lookup_grad/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
ô
&gradients/embedding_lookup_grad/concatConcatV2*gradients/embedding_lookup_grad/ExpandDims-gradients/embedding_lookup_grad/strided_slice+gradients/embedding_lookup_grad/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
­
'gradients/embedding_lookup_grad/ReshapeReshapegradients/AddN_8&gradients/embedding_lookup_grad/concat*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
Š
)gradients/embedding_lookup_grad/Reshape_1ReshapePlaceholder*gradients/embedding_lookup_grad/ExpandDims*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
w
beta1_power/initial_valueConst*
dtype0*
_class
	loc:@Bias*
valueB
 *fff?*
_output_shapes
: 

beta1_power
VariableV2*
_class
	loc:@Bias*
_output_shapes
: *
shared_name *
	container *
dtype0*
shape: 
§
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
validate_shape(*
_class
	loc:@Bias*
T0*
use_locking(*
_output_shapes
: 
c
beta1_power/readIdentitybeta1_power*
T0*
_class
	loc:@Bias*
_output_shapes
: 
w
beta2_power/initial_valueConst*
_class
	loc:@Bias*
dtype0*
_output_shapes
: *
valueB
 *wž?

beta2_power
VariableV2*
shape: *
dtype0*
shared_name *
	container *
_output_shapes
: *
_class
	loc:@Bias
§
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
T0*
validate_shape(*
use_locking(*
_class
	loc:@Bias*
_output_shapes
: 
c
beta2_power/readIdentitybeta2_power*
_output_shapes
: *
_class
	loc:@Bias*
T0

0Embedding/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_class
loc:@Embedding*
_output_shapes
:*
valueB"      

&Embedding/Adam/Initializer/zeros/ConstConst*
_class
loc:@Embedding*
valueB
 *    *
dtype0*
_output_shapes
: 
Ü
 Embedding/Adam/Initializer/zerosFill0Embedding/Adam/Initializer/zeros/shape_as_tensor&Embedding/Adam/Initializer/zeros/Const*
T0*
_class
loc:@Embedding*

index_type0*
_output_shapes
:	
˘
Embedding/Adam
VariableV2*
	container *
shared_name *
_output_shapes
:	*
_class
loc:@Embedding*
dtype0*
shape:	
Â
Embedding/Adam/AssignAssignEmbedding/Adam Embedding/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes
:	*
_class
loc:@Embedding*
T0
w
Embedding/Adam/readIdentityEmbedding/Adam*
_class
loc:@Embedding*
T0*
_output_shapes
:	
Ą
2Embedding/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
_class
loc:@Embedding*
valueB"      *
dtype0

(Embedding/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    *
_class
loc:@Embedding
â
"Embedding/Adam_1/Initializer/zerosFill2Embedding/Adam_1/Initializer/zeros/shape_as_tensor(Embedding/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Embedding*
_output_shapes
:	
¤
Embedding/Adam_1
VariableV2*
_output_shapes
:	*
dtype0*
shared_name *
_class
loc:@Embedding*
shape:	*
	container 
Č
Embedding/Adam_1/AssignAssignEmbedding/Adam_1"Embedding/Adam_1/Initializer/zeros*
T0*
use_locking(*
validate_shape(*
_output_shapes
:	*
_class
loc:@Embedding
{
Embedding/Adam_1/readIdentityEmbedding/Adam_1*
_output_shapes
:	*
_class
loc:@Embedding*
T0
ă
Rbidirectional_rnn/fw/basic_lstm_cell/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*>
_class4
20loc:@bidirectional_rnn/fw/basic_lstm_cell/kernel*
valueB"@     *
dtype0
Í
Hbidirectional_rnn/fw/basic_lstm_cell/kernel/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *>
_class4
20loc:@bidirectional_rnn/fw/basic_lstm_cell/kernel*
valueB
 *    *
dtype0
ĺ
Bbidirectional_rnn/fw/basic_lstm_cell/kernel/Adam/Initializer/zerosFillRbidirectional_rnn/fw/basic_lstm_cell/kernel/Adam/Initializer/zeros/shape_as_tensorHbidirectional_rnn/fw/basic_lstm_cell/kernel/Adam/Initializer/zeros/Const* 
_output_shapes
:
Ŕ*
T0*>
_class4
20loc:@bidirectional_rnn/fw/basic_lstm_cell/kernel*

index_type0
č
0bidirectional_rnn/fw/basic_lstm_cell/kernel/Adam
VariableV2*
dtype0* 
_output_shapes
:
Ŕ*>
_class4
20loc:@bidirectional_rnn/fw/basic_lstm_cell/kernel*
	container *
shared_name *
shape:
Ŕ
Ë
7bidirectional_rnn/fw/basic_lstm_cell/kernel/Adam/AssignAssign0bidirectional_rnn/fw/basic_lstm_cell/kernel/AdamBbidirectional_rnn/fw/basic_lstm_cell/kernel/Adam/Initializer/zeros*
T0*>
_class4
20loc:@bidirectional_rnn/fw/basic_lstm_cell/kernel*
validate_shape(*
use_locking(* 
_output_shapes
:
Ŕ
Ţ
5bidirectional_rnn/fw/basic_lstm_cell/kernel/Adam/readIdentity0bidirectional_rnn/fw/basic_lstm_cell/kernel/Adam*>
_class4
20loc:@bidirectional_rnn/fw/basic_lstm_cell/kernel*
T0* 
_output_shapes
:
Ŕ
ĺ
Tbidirectional_rnn/fw/basic_lstm_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"@     *
_output_shapes
:*>
_class4
20loc:@bidirectional_rnn/fw/basic_lstm_cell/kernel*
dtype0
Ď
Jbidirectional_rnn/fw/basic_lstm_cell/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: *>
_class4
20loc:@bidirectional_rnn/fw/basic_lstm_cell/kernel
ë
Dbidirectional_rnn/fw/basic_lstm_cell/kernel/Adam_1/Initializer/zerosFillTbidirectional_rnn/fw/basic_lstm_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensorJbidirectional_rnn/fw/basic_lstm_cell/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
Ŕ*

index_type0*
T0*>
_class4
20loc:@bidirectional_rnn/fw/basic_lstm_cell/kernel
ę
2bidirectional_rnn/fw/basic_lstm_cell/kernel/Adam_1
VariableV2*
shared_name *
	container *>
_class4
20loc:@bidirectional_rnn/fw/basic_lstm_cell/kernel* 
_output_shapes
:
Ŕ*
dtype0*
shape:
Ŕ
Ń
9bidirectional_rnn/fw/basic_lstm_cell/kernel/Adam_1/AssignAssign2bidirectional_rnn/fw/basic_lstm_cell/kernel/Adam_1Dbidirectional_rnn/fw/basic_lstm_cell/kernel/Adam_1/Initializer/zeros*>
_class4
20loc:@bidirectional_rnn/fw/basic_lstm_cell/kernel*
validate_shape(*
T0*
use_locking(* 
_output_shapes
:
Ŕ
â
7bidirectional_rnn/fw/basic_lstm_cell/kernel/Adam_1/readIdentity2bidirectional_rnn/fw/basic_lstm_cell/kernel/Adam_1*>
_class4
20loc:@bidirectional_rnn/fw/basic_lstm_cell/kernel* 
_output_shapes
:
Ŕ*
T0
Í
@bidirectional_rnn/fw/basic_lstm_cell/bias/Adam/Initializer/zerosConst*
valueB*    *
_output_shapes	
:*<
_class2
0.loc:@bidirectional_rnn/fw/basic_lstm_cell/bias*
dtype0
Ú
.bidirectional_rnn/fw/basic_lstm_cell/bias/Adam
VariableV2*
_output_shapes	
:*<
_class2
0.loc:@bidirectional_rnn/fw/basic_lstm_cell/bias*
	container *
dtype0*
shared_name *
shape:
ž
5bidirectional_rnn/fw/basic_lstm_cell/bias/Adam/AssignAssign.bidirectional_rnn/fw/basic_lstm_cell/bias/Adam@bidirectional_rnn/fw/basic_lstm_cell/bias/Adam/Initializer/zeros*<
_class2
0.loc:@bidirectional_rnn/fw/basic_lstm_cell/bias*
T0*
validate_shape(*
_output_shapes	
:*
use_locking(
Ó
3bidirectional_rnn/fw/basic_lstm_cell/bias/Adam/readIdentity.bidirectional_rnn/fw/basic_lstm_cell/bias/Adam*<
_class2
0.loc:@bidirectional_rnn/fw/basic_lstm_cell/bias*
T0*
_output_shapes	
:
Ď
Bbidirectional_rnn/fw/basic_lstm_cell/bias/Adam_1/Initializer/zerosConst*
valueB*    *
_output_shapes	
:*
dtype0*<
_class2
0.loc:@bidirectional_rnn/fw/basic_lstm_cell/bias
Ü
0bidirectional_rnn/fw/basic_lstm_cell/bias/Adam_1
VariableV2*
_output_shapes	
:*<
_class2
0.loc:@bidirectional_rnn/fw/basic_lstm_cell/bias*
shared_name *
	container *
dtype0*
shape:
Ä
7bidirectional_rnn/fw/basic_lstm_cell/bias/Adam_1/AssignAssign0bidirectional_rnn/fw/basic_lstm_cell/bias/Adam_1Bbidirectional_rnn/fw/basic_lstm_cell/bias/Adam_1/Initializer/zeros*
validate_shape(*
T0*<
_class2
0.loc:@bidirectional_rnn/fw/basic_lstm_cell/bias*
use_locking(*
_output_shapes	
:
×
5bidirectional_rnn/fw/basic_lstm_cell/bias/Adam_1/readIdentity0bidirectional_rnn/fw/basic_lstm_cell/bias/Adam_1*
_output_shapes	
:*<
_class2
0.loc:@bidirectional_rnn/fw/basic_lstm_cell/bias*
T0
ă
Rbidirectional_rnn/bw/basic_lstm_cell/kernel/Adam/Initializer/zeros/shape_as_tensorConst*>
_class4
20loc:@bidirectional_rnn/bw/basic_lstm_cell/kernel*
dtype0*
valueB"@     *
_output_shapes
:
Í
Hbidirectional_rnn/bw/basic_lstm_cell/kernel/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *>
_class4
20loc:@bidirectional_rnn/bw/basic_lstm_cell/kernel*
valueB
 *    *
dtype0
ĺ
Bbidirectional_rnn/bw/basic_lstm_cell/kernel/Adam/Initializer/zerosFillRbidirectional_rnn/bw/basic_lstm_cell/kernel/Adam/Initializer/zeros/shape_as_tensorHbidirectional_rnn/bw/basic_lstm_cell/kernel/Adam/Initializer/zeros/Const*>
_class4
20loc:@bidirectional_rnn/bw/basic_lstm_cell/kernel*
T0* 
_output_shapes
:
Ŕ*

index_type0
č
0bidirectional_rnn/bw/basic_lstm_cell/kernel/Adam
VariableV2*
dtype0*
shape:
Ŕ*
shared_name *
	container * 
_output_shapes
:
Ŕ*>
_class4
20loc:@bidirectional_rnn/bw/basic_lstm_cell/kernel
Ë
7bidirectional_rnn/bw/basic_lstm_cell/kernel/Adam/AssignAssign0bidirectional_rnn/bw/basic_lstm_cell/kernel/AdamBbidirectional_rnn/bw/basic_lstm_cell/kernel/Adam/Initializer/zeros*>
_class4
20loc:@bidirectional_rnn/bw/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
Ŕ*
use_locking(*
T0
Ţ
5bidirectional_rnn/bw/basic_lstm_cell/kernel/Adam/readIdentity0bidirectional_rnn/bw/basic_lstm_cell/kernel/Adam* 
_output_shapes
:
Ŕ*
T0*>
_class4
20loc:@bidirectional_rnn/bw/basic_lstm_cell/kernel
ĺ
Tbidirectional_rnn/bw/basic_lstm_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*>
_class4
20loc:@bidirectional_rnn/bw/basic_lstm_cell/kernel*
_output_shapes
:*
valueB"@     *
dtype0
Ď
Jbidirectional_rnn/bw/basic_lstm_cell/kernel/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *>
_class4
20loc:@bidirectional_rnn/bw/basic_lstm_cell/kernel*
valueB
 *    *
dtype0
ë
Dbidirectional_rnn/bw/basic_lstm_cell/kernel/Adam_1/Initializer/zerosFillTbidirectional_rnn/bw/basic_lstm_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensorJbidirectional_rnn/bw/basic_lstm_cell/kernel/Adam_1/Initializer/zeros/Const*>
_class4
20loc:@bidirectional_rnn/bw/basic_lstm_cell/kernel* 
_output_shapes
:
Ŕ*
T0*

index_type0
ę
2bidirectional_rnn/bw/basic_lstm_cell/kernel/Adam_1
VariableV2*
dtype0*
	container *
shared_name *
shape:
Ŕ* 
_output_shapes
:
Ŕ*>
_class4
20loc:@bidirectional_rnn/bw/basic_lstm_cell/kernel
Ń
9bidirectional_rnn/bw/basic_lstm_cell/kernel/Adam_1/AssignAssign2bidirectional_rnn/bw/basic_lstm_cell/kernel/Adam_1Dbidirectional_rnn/bw/basic_lstm_cell/kernel/Adam_1/Initializer/zeros* 
_output_shapes
:
Ŕ*
T0*
validate_shape(*>
_class4
20loc:@bidirectional_rnn/bw/basic_lstm_cell/kernel*
use_locking(
â
7bidirectional_rnn/bw/basic_lstm_cell/kernel/Adam_1/readIdentity2bidirectional_rnn/bw/basic_lstm_cell/kernel/Adam_1*>
_class4
20loc:@bidirectional_rnn/bw/basic_lstm_cell/kernel* 
_output_shapes
:
Ŕ*
T0
Í
@bidirectional_rnn/bw/basic_lstm_cell/bias/Adam/Initializer/zerosConst*<
_class2
0.loc:@bidirectional_rnn/bw/basic_lstm_cell/bias*
valueB*    *
_output_shapes	
:*
dtype0
Ú
.bidirectional_rnn/bw/basic_lstm_cell/bias/Adam
VariableV2*
shape:*
_output_shapes	
:*
dtype0*<
_class2
0.loc:@bidirectional_rnn/bw/basic_lstm_cell/bias*
shared_name *
	container 
ž
5bidirectional_rnn/bw/basic_lstm_cell/bias/Adam/AssignAssign.bidirectional_rnn/bw/basic_lstm_cell/bias/Adam@bidirectional_rnn/bw/basic_lstm_cell/bias/Adam/Initializer/zeros*
use_locking(*
T0*
_output_shapes	
:*<
_class2
0.loc:@bidirectional_rnn/bw/basic_lstm_cell/bias*
validate_shape(
Ó
3bidirectional_rnn/bw/basic_lstm_cell/bias/Adam/readIdentity.bidirectional_rnn/bw/basic_lstm_cell/bias/Adam*<
_class2
0.loc:@bidirectional_rnn/bw/basic_lstm_cell/bias*
_output_shapes	
:*
T0
Ď
Bbidirectional_rnn/bw/basic_lstm_cell/bias/Adam_1/Initializer/zerosConst*
valueB*    *<
_class2
0.loc:@bidirectional_rnn/bw/basic_lstm_cell/bias*
_output_shapes	
:*
dtype0
Ü
0bidirectional_rnn/bw/basic_lstm_cell/bias/Adam_1
VariableV2*<
_class2
0.loc:@bidirectional_rnn/bw/basic_lstm_cell/bias*
shared_name *
shape:*
dtype0*
_output_shapes	
:*
	container 
Ä
7bidirectional_rnn/bw/basic_lstm_cell/bias/Adam_1/AssignAssign0bidirectional_rnn/bw/basic_lstm_cell/bias/Adam_1Bbidirectional_rnn/bw/basic_lstm_cell/bias/Adam_1/Initializer/zeros*
_output_shapes	
:*<
_class2
0.loc:@bidirectional_rnn/bw/basic_lstm_cell/bias*
use_locking(*
validate_shape(*
T0
×
5bidirectional_rnn/bw/basic_lstm_cell/bias/Adam_1/readIdentity0bidirectional_rnn/bw/basic_lstm_cell/bias/Adam_1*
T0*<
_class2
0.loc:@bidirectional_rnn/bw/basic_lstm_cell/bias*
_output_shapes	
:

Weights/Adam/Initializer/zerosConst*
dtype0*
_class
loc:@Weights*
_output_shapes
:	*
valueB	*    

Weights/Adam
VariableV2*
_output_shapes
:	*
	container *
shared_name *
_class
loc:@Weights*
dtype0*
shape:	
ş
Weights/Adam/AssignAssignWeights/AdamWeights/Adam/Initializer/zeros*
use_locking(*
_class
loc:@Weights*
_output_shapes
:	*
T0*
validate_shape(
q
Weights/Adam/readIdentityWeights/Adam*
T0*
_output_shapes
:	*
_class
loc:@Weights

 Weights/Adam_1/Initializer/zerosConst*
dtype0*
_class
loc:@Weights*
valueB	*    *
_output_shapes
:	
 
Weights/Adam_1
VariableV2*
dtype0*
	container *
_output_shapes
:	*
_class
loc:@Weights*
shared_name *
shape:	
Ŕ
Weights/Adam_1/AssignAssignWeights/Adam_1 Weights/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Weights*
validate_shape(*
_output_shapes
:	
u
Weights/Adam_1/readIdentityWeights/Adam_1*
T0*
_class
loc:@Weights*
_output_shapes
:	

Bias/Adam/Initializer/zerosConst*
_class
	loc:@Bias*
_output_shapes
:*
dtype0*
valueB*    

	Bias/Adam
VariableV2*
	container *
dtype0*
shared_name *
shape:*
_output_shapes
:*
_class
	loc:@Bias
Š
Bias/Adam/AssignAssign	Bias/AdamBias/Adam/Initializer/zeros*
validate_shape(*
T0*
_output_shapes
:*
_class
	loc:@Bias*
use_locking(
c
Bias/Adam/readIdentity	Bias/Adam*
_output_shapes
:*
_class
	loc:@Bias*
T0

Bias/Adam_1/Initializer/zerosConst*
valueB*    *
_output_shapes
:*
_class
	loc:@Bias*
dtype0

Bias/Adam_1
VariableV2*
shape:*
dtype0*
_output_shapes
:*
_class
	loc:@Bias*
shared_name *
	container 
Ż
Bias/Adam_1/AssignAssignBias/Adam_1Bias/Adam_1/Initializer/zeros*
validate_shape(*
_class
	loc:@Bias*
use_locking(*
T0*
_output_shapes
:
g
Bias/Adam_1/readIdentityBias/Adam_1*
_class
	loc:@Bias*
T0*
_output_shapes
:
W
Adam/learning_rateConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
O

Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
O

Adam/beta2Const*
_output_shapes
: *
dtype0*
valueB
 *wž?
Q
Adam/epsilonConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+2
ť
Adam/update_Embedding/UniqueUnique)gradients/embedding_lookup_grad/Reshape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
out_idx0*
_class
loc:@Embedding

Adam/update_Embedding/ShapeShapeAdam/update_Embedding/Unique*
out_type0*
T0*
_output_shapes
:*
_class
loc:@Embedding

)Adam/update_Embedding/strided_slice/stackConst*
_class
loc:@Embedding*
valueB: *
dtype0*
_output_shapes
:

+Adam/update_Embedding/strided_slice/stack_1Const*
dtype0*
_class
loc:@Embedding*
valueB:*
_output_shapes
:

+Adam/update_Embedding/strided_slice/stack_2Const*
valueB:*
dtype0*
_class
loc:@Embedding*
_output_shapes
:

#Adam/update_Embedding/strided_sliceStridedSliceAdam/update_Embedding/Shape)Adam/update_Embedding/strided_slice/stack+Adam/update_Embedding/strided_slice/stack_1+Adam/update_Embedding/strided_slice/stack_2*
_class
loc:@Embedding*
_output_shapes
: *

begin_mask *
T0*
shrink_axis_mask*
ellipsis_mask *
Index0*
end_mask *
new_axis_mask 
Ą
(Adam/update_Embedding/UnsortedSegmentSumUnsortedSegmentSum'gradients/embedding_lookup_grad/ReshapeAdam/update_Embedding/Unique:1#Adam/update_Embedding/strided_slice*
Tnumsegments0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
_class
loc:@Embedding*
Tindices0*
T0
~
Adam/update_Embedding/sub/xConst*
_output_shapes
: *
dtype0*
_class
loc:@Embedding*
valueB
 *  ?

Adam/update_Embedding/subSubAdam/update_Embedding/sub/xbeta2_power/read*
_class
loc:@Embedding*
_output_shapes
: *
T0
|
Adam/update_Embedding/SqrtSqrtAdam/update_Embedding/sub*
T0*
_output_shapes
: *
_class
loc:@Embedding

Adam/update_Embedding/mulMulAdam/learning_rateAdam/update_Embedding/Sqrt*
_class
loc:@Embedding*
T0*
_output_shapes
: 

Adam/update_Embedding/sub_1/xConst*
dtype0*
_class
loc:@Embedding*
_output_shapes
: *
valueB
 *  ?

Adam/update_Embedding/sub_1SubAdam/update_Embedding/sub_1/xbeta1_power/read*
_output_shapes
: *
T0*
_class
loc:@Embedding

Adam/update_Embedding/truedivRealDivAdam/update_Embedding/mulAdam/update_Embedding/sub_1*
_class
loc:@Embedding*
_output_shapes
: *
T0

Adam/update_Embedding/sub_2/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?*
_class
loc:@Embedding

Adam/update_Embedding/sub_2SubAdam/update_Embedding/sub_2/x
Adam/beta1*
T0*
_class
loc:@Embedding*
_output_shapes
: 
ş
Adam/update_Embedding/mul_1Mul(Adam/update_Embedding/UnsortedSegmentSumAdam/update_Embedding/sub_2*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
_class
loc:@Embedding

Adam/update_Embedding/mul_2MulEmbedding/Adam/read
Adam/beta1*
_class
loc:@Embedding*
_output_shapes
:	*
T0
Ä
Adam/update_Embedding/AssignAssignEmbedding/AdamAdam/update_Embedding/mul_2*
_class
loc:@Embedding*
T0*
validate_shape(*
use_locking( *
_output_shapes
:	

 Adam/update_Embedding/ScatterAdd
ScatterAddEmbedding/AdamAdam/update_Embedding/UniqueAdam/update_Embedding/mul_1^Adam/update_Embedding/Assign*
T0*
_class
loc:@Embedding*
_output_shapes
:	*
Tindices0*
use_locking( 
Ç
Adam/update_Embedding/mul_3Mul(Adam/update_Embedding/UnsortedSegmentSum(Adam/update_Embedding/UnsortedSegmentSum*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
_class
loc:@Embedding

Adam/update_Embedding/sub_3/xConst*
_class
loc:@Embedding*
valueB
 *  ?*
_output_shapes
: *
dtype0

Adam/update_Embedding/sub_3SubAdam/update_Embedding/sub_3/x
Adam/beta2*
_output_shapes
: *
T0*
_class
loc:@Embedding
­
Adam/update_Embedding/mul_4MulAdam/update_Embedding/mul_3Adam/update_Embedding/sub_3*
T0*
_class
loc:@Embedding*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Adam/update_Embedding/mul_5MulEmbedding/Adam_1/read
Adam/beta2*
T0*
_output_shapes
:	*
_class
loc:@Embedding
Č
Adam/update_Embedding/Assign_1AssignEmbedding/Adam_1Adam/update_Embedding/mul_5*
_output_shapes
:	*
T0*
validate_shape(*
_class
loc:@Embedding*
use_locking( 

"Adam/update_Embedding/ScatterAdd_1
ScatterAddEmbedding/Adam_1Adam/update_Embedding/UniqueAdam/update_Embedding/mul_4^Adam/update_Embedding/Assign_1*
_output_shapes
:	*
T0*
_class
loc:@Embedding*
use_locking( *
Tindices0

Adam/update_Embedding/Sqrt_1Sqrt"Adam/update_Embedding/ScatterAdd_1*
_class
loc:@Embedding*
T0*
_output_shapes
:	
Ť
Adam/update_Embedding/mul_6MulAdam/update_Embedding/truediv Adam/update_Embedding/ScatterAdd*
_output_shapes
:	*
T0*
_class
loc:@Embedding

Adam/update_Embedding/addAddAdam/update_Embedding/Sqrt_1Adam/epsilon*
T0*
_output_shapes
:	*
_class
loc:@Embedding
Ş
Adam/update_Embedding/truediv_1RealDivAdam/update_Embedding/mul_6Adam/update_Embedding/add*
_class
loc:@Embedding*
_output_shapes
:	*
T0
ł
Adam/update_Embedding/AssignSub	AssignSub	EmbeddingAdam/update_Embedding/truediv_1*
T0*
_class
loc:@Embedding*
use_locking( *
_output_shapes
:	
°
 Adam/update_Embedding/group_depsNoOp ^Adam/update_Embedding/AssignSub!^Adam/update_Embedding/ScatterAdd#^Adam/update_Embedding/ScatterAdd_1*
_class
loc:@Embedding
¤
AAdam/update_bidirectional_rnn/fw/basic_lstm_cell/kernel/ApplyAdam	ApplyAdam+bidirectional_rnn/fw/basic_lstm_cell/kernel0bidirectional_rnn/fw/basic_lstm_cell/kernel/Adam2bidirectional_rnn/fw/basic_lstm_cell/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonQgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_3*
use_nesterov( *
use_locking( * 
_output_shapes
:
Ŕ*
T0*>
_class4
20loc:@bidirectional_rnn/fw/basic_lstm_cell/kernel

?Adam/update_bidirectional_rnn/fw/basic_lstm_cell/bias/ApplyAdam	ApplyAdam)bidirectional_rnn/fw/basic_lstm_cell/bias.bidirectional_rnn/fw/basic_lstm_cell/bias/Adam0bidirectional_rnn/fw/basic_lstm_cell/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonRgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3*<
_class2
0.loc:@bidirectional_rnn/fw/basic_lstm_cell/bias*
use_nesterov( *
_output_shapes	
:*
use_locking( *
T0
¤
AAdam/update_bidirectional_rnn/bw/basic_lstm_cell/kernel/ApplyAdam	ApplyAdam+bidirectional_rnn/bw/basic_lstm_cell/kernel0bidirectional_rnn/bw/basic_lstm_cell/kernel/Adam2bidirectional_rnn/bw/basic_lstm_cell/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonQgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_3*>
_class4
20loc:@bidirectional_rnn/bw/basic_lstm_cell/kernel*
use_nesterov( *
T0*
use_locking( * 
_output_shapes
:
Ŕ

?Adam/update_bidirectional_rnn/bw/basic_lstm_cell/bias/ApplyAdam	ApplyAdam)bidirectional_rnn/bw/basic_lstm_cell/bias.bidirectional_rnn/bw/basic_lstm_cell/bias/Adam0bidirectional_rnn/bw/basic_lstm_cell/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonRgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3*
use_locking( *
_output_shapes	
:*<
_class2
0.loc:@bidirectional_rnn/bw/basic_lstm_cell/bias*
T0*
use_nesterov( 
Î
Adam/update_Weights/ApplyAdam	ApplyAdamWeightsWeights/AdamWeights/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_output_shapes
:	*
use_nesterov( *
_class
loc:@Weights
ˇ
Adam/update_Bias/ApplyAdam	ApplyAdamBias	Bias/AdamBias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon-gradients/add_grad/tuple/control_dependency_1*
T0*
_class
	loc:@Bias*
use_nesterov( *
_output_shapes
:*
use_locking( 
Ó
Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_Bias/ApplyAdam!^Adam/update_Embedding/group_deps^Adam/update_Weights/ApplyAdam@^Adam/update_bidirectional_rnn/bw/basic_lstm_cell/bias/ApplyAdamB^Adam/update_bidirectional_rnn/bw/basic_lstm_cell/kernel/ApplyAdam@^Adam/update_bidirectional_rnn/fw/basic_lstm_cell/bias/ApplyAdamB^Adam/update_bidirectional_rnn/fw/basic_lstm_cell/kernel/ApplyAdam*
T0*
_output_shapes
: *
_class
	loc:@Bias

Adam/AssignAssignbeta1_powerAdam/mul*
validate_shape(*
use_locking( *
_output_shapes
: *
T0*
_class
	loc:@Bias
Ő

Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_Bias/ApplyAdam!^Adam/update_Embedding/group_deps^Adam/update_Weights/ApplyAdam@^Adam/update_bidirectional_rnn/bw/basic_lstm_cell/bias/ApplyAdamB^Adam/update_bidirectional_rnn/bw/basic_lstm_cell/kernel/ApplyAdam@^Adam/update_bidirectional_rnn/fw/basic_lstm_cell/bias/ApplyAdamB^Adam/update_bidirectional_rnn/fw/basic_lstm_cell/kernel/ApplyAdam*
T0*
_class
	loc:@Bias*
_output_shapes
: 

Adam/Assign_1Assignbeta2_power
Adam/mul_1*
T0*
_output_shapes
: *
_class
	loc:@Bias*
validate_shape(*
use_locking( 

AdamNoOp^Adam/Assign^Adam/Assign_1^Adam/update_Bias/ApplyAdam!^Adam/update_Embedding/group_deps^Adam/update_Weights/ApplyAdam@^Adam/update_bidirectional_rnn/bw/basic_lstm_cell/bias/ApplyAdamB^Adam/update_bidirectional_rnn/bw/basic_lstm_cell/kernel/ApplyAdam@^Adam/update_bidirectional_rnn/fw/basic_lstm_cell/bias/ApplyAdamB^Adam/update_bidirectional_rnn/fw/basic_lstm_cell/kernel/ApplyAdam

initNoOp^Bias/Adam/Assign^Bias/Adam_1/Assign^Bias/Assign^Embedding/Adam/Assign^Embedding/Adam_1/Assign^Embedding/Assign^Weights/Adam/Assign^Weights/Adam_1/Assign^Weights/Assign^beta1_power/Assign^beta2_power/Assign6^bidirectional_rnn/bw/basic_lstm_cell/bias/Adam/Assign8^bidirectional_rnn/bw/basic_lstm_cell/bias/Adam_1/Assign1^bidirectional_rnn/bw/basic_lstm_cell/bias/Assign8^bidirectional_rnn/bw/basic_lstm_cell/kernel/Adam/Assign:^bidirectional_rnn/bw/basic_lstm_cell/kernel/Adam_1/Assign3^bidirectional_rnn/bw/basic_lstm_cell/kernel/Assign6^bidirectional_rnn/fw/basic_lstm_cell/bias/Adam/Assign8^bidirectional_rnn/fw/basic_lstm_cell/bias/Adam_1/Assign1^bidirectional_rnn/fw/basic_lstm_cell/bias/Assign8^bidirectional_rnn/fw/basic_lstm_cell/kernel/Adam/Assign:^bidirectional_rnn/fw/basic_lstm_cell/kernel/Adam_1/Assign3^bidirectional_rnn/fw/basic_lstm_cell/kernel/Assign
Y
save/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
n
save/filenamePlaceholderWithDefaultsave/filename/input*
shape: *
_output_shapes
: *
dtype0
e

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
_output_shapes
: *
shape: 
Ź
save/SaveV2/tensor_namesConst*ß
valueŐBŇBBiasB	EmbeddingBWeightsB)bidirectional_rnn/bw/basic_lstm_cell/biasB+bidirectional_rnn/bw/basic_lstm_cell/kernelB)bidirectional_rnn/fw/basic_lstm_cell/biasB+bidirectional_rnn/fw/basic_lstm_cell/kernel*
dtype0*
_output_shapes
:
q
save/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:*!
valueBB B B B B B B 
ş
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesBias	EmbeddingWeights)bidirectional_rnn/bw/basic_lstm_cell/bias+bidirectional_rnn/bw/basic_lstm_cell/kernel)bidirectional_rnn/fw/basic_lstm_cell/bias+bidirectional_rnn/fw/basic_lstm_cell/kernel*
dtypes
	2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_output_shapes
: *
_class
loc:@save/Const
ž
save/RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ß
valueŐBŇBBiasB	EmbeddingBWeightsB)bidirectional_rnn/bw/basic_lstm_cell/biasB+bidirectional_rnn/bw/basic_lstm_cell/kernelB)bidirectional_rnn/fw/basic_lstm_cell/biasB+bidirectional_rnn/fw/basic_lstm_cell/kernel

save/RestoreV2/shape_and_slicesConst"/device:CPU:0*!
valueBB B B B B B B *
_output_shapes
:*
dtype0
˝
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
	2*0
_output_shapes
:::::::

save/AssignAssignBiassave/RestoreV2*
T0*
_output_shapes
:*
_class
	loc:@Bias*
use_locking(*
validate_shape(
Ľ
save/Assign_1Assign	Embeddingsave/RestoreV2:1*
T0*
_output_shapes
:	*
validate_shape(*
_class
loc:@Embedding*
use_locking(
Ą
save/Assign_2AssignWeightssave/RestoreV2:2*
_output_shapes
:	*
T0*
validate_shape(*
_class
loc:@Weights*
use_locking(
á
save/Assign_3Assign)bidirectional_rnn/bw/basic_lstm_cell/biassave/RestoreV2:3*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:*<
_class2
0.loc:@bidirectional_rnn/bw/basic_lstm_cell/bias
ę
save/Assign_4Assign+bidirectional_rnn/bw/basic_lstm_cell/kernelsave/RestoreV2:4* 
_output_shapes
:
Ŕ*
use_locking(*
validate_shape(*
T0*>
_class4
20loc:@bidirectional_rnn/bw/basic_lstm_cell/kernel
á
save/Assign_5Assign)bidirectional_rnn/fw/basic_lstm_cell/biassave/RestoreV2:5*
_output_shapes	
:*
validate_shape(*
use_locking(*
T0*<
_class2
0.loc:@bidirectional_rnn/fw/basic_lstm_cell/bias
ę
save/Assign_6Assign+bidirectional_rnn/fw/basic_lstm_cell/kernelsave/RestoreV2:6*
validate_shape(* 
_output_shapes
:
Ŕ*
T0*>
_class4
20loc:@bidirectional_rnn/fw/basic_lstm_cell/kernel*
use_locking(

save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6

init_1NoOp^Bias/Adam/Assign^Bias/Adam_1/Assign^Bias/Assign^Embedding/Adam/Assign^Embedding/Adam_1/Assign^Embedding/Assign^Weights/Adam/Assign^Weights/Adam_1/Assign^Weights/Assign^beta1_power/Assign^beta2_power/Assign6^bidirectional_rnn/bw/basic_lstm_cell/bias/Adam/Assign8^bidirectional_rnn/bw/basic_lstm_cell/bias/Adam_1/Assign1^bidirectional_rnn/bw/basic_lstm_cell/bias/Assign8^bidirectional_rnn/bw/basic_lstm_cell/kernel/Adam/Assign:^bidirectional_rnn/bw/basic_lstm_cell/kernel/Adam_1/Assign3^bidirectional_rnn/bw/basic_lstm_cell/kernel/Assign6^bidirectional_rnn/fw/basic_lstm_cell/bias/Adam/Assign8^bidirectional_rnn/fw/basic_lstm_cell/bias/Adam_1/Assign1^bidirectional_rnn/fw/basic_lstm_cell/bias/Assign8^bidirectional_rnn/fw/basic_lstm_cell/kernel/Adam/Assign:^bidirectional_rnn/fw/basic_lstm_cell/kernel/Adam_1/Assign3^bidirectional_rnn/fw/basic_lstm_cell/kernel/Assign"&(ĽSÔźó     y×	 tţŘAJŻç
Ś>˙=
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	
h
All	
input

reduction_indices"Tidx

output
"
	keep_dimsbool( "
Tidxtype0:
2	
î
	ApplyAdam
var"T	
m"T	
v"T
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 

ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
P
Assert
	condition
	
data2T"
T
list(type)(0"
	summarizeint
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
s
	AssignSub
ref"T

value"T

output_ref"T" 
Ttype:
2	"
use_lockingbool( 
k
BatchMatMulV2
x"T
y"T
output"T"
Ttype:

2	"
adj_xbool( "
adj_ybool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
I
ConcatOffset

concat_dim
shape*N
offset*N"
Nint(0
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype

ControlTrigger
y
Enter	
data"T
output"T"	
Ttype"

frame_namestring"
is_constantbool( "
parallel_iterationsint

B
Equal
x"T
y"T
z
"
Ttype:
2	

)
Exit	
data"T
output"T"	
Ttype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
­
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
:
InvertPermutation
x"T
y"T"
Ttype0:
2	
:
Less
x"T
y"T
z
"
Ttype:
2	
?

LogSoftmax
logits"T

logsoftmax"T"
Ttype:
2
$

LogicalAnd
x

y

z

!
LoopCond	
input


output

q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
8
Maximum
x"T
y"T
z"T"
Ttype:

2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0

Min

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
8
Minimum
x"T
y"T
z"T"
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	
.
Neg
x"T
y"T"
Ttype:

2	
2
NextIteration	
data"T
output"T"	
Ttype

NoOp

OneHot
indices"TI	
depth
on_value"T
	off_value"T
output"T"
axisint˙˙˙˙˙˙˙˙˙"	
Ttype"
TItype0	:
2	
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
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
)
Rank

input"T

output"	
Ttype
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
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
list(type)(0

ReverseSequence

input"T
seq_lengths"Tlen
output"T"
seq_dimint"
	batch_dimint "	
Ttype"
Tlentype0	:
2	
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
Ľ

ScatterAdd
ref"T
indices"Tindices
updates"T

output_ref"T" 
Ttype:
2	"
Tindicestype:
2	"
use_lockingbool( 
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
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
0
Sigmoid
x"T
y"T"
Ttype:

2
=
SigmoidGrad
y"T
dy"T
z"T"
Ttype:

2
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
j
SoftmaxCrossEntropyWithLogits
features"T
labels"T	
loss"T
backprop"T"
Ttype:
2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
-
Sqrt
x"T
y"T"
Ttype:

2
A

StackPopV2

handle
elem"	elem_type"
	elem_typetype
X
StackPushV2

handle	
elem"T
output"T"	
Ttype"
swap_memorybool( 
S
StackV2
max_size

handle"
	elem_typetype"

stack_namestring 
2
StopGradient

input"T
output"T"	
Ttype
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
:
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
-
Tanh
x"T
y"T"
Ttype:

2
:
TanhGrad
y"T
dy"T
z"T"
Ttype:

2
{
TensorArrayGatherV3

handle
indices
flow_in
value"dtype"
dtypetype"
element_shapeshape:
`
TensorArrayGradV3

handle
flow_in
grad_handle
flow_out"
sourcestring
Y
TensorArrayReadV3

handle	
index
flow_in
value"dtype"
dtypetype
d
TensorArrayScatterV3

handle
indices

value"T
flow_in
flow_out"	
Ttype
9
TensorArraySizeV3

handle
flow_in
size
Ţ
TensorArrayV3
size

handle
flow"
dtypetype"
element_shapeshape:"
dynamic_sizebool( "
clear_after_readbool("$
identical_element_shapesbool( "
tensor_array_namestring 
`
TensorArrayWriteV3

handle	
index

value"T
flow_in
flow_out"	
Ttype
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
P
Unique
x"T
y"T
idx"out_idx"	
Ttype"
out_idxtype0:
2	
Á
UnsortedSegmentSum	
data"T
segment_ids"Tindices
num_segments"Tnumsegments
output"T" 
Ttype:
2	"
Tindicestype:
2	" 
Tnumsegmentstype0:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring 
&
	ZerosLike
x"T
y"T"	
Ttype*1.14.02v1.14.0-rc1-22-gaf24dc91b5íŔ
p
PlaceholderPlaceholder*
shape:˙˙˙˙˙˙˙˙˙*
dtype0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
R
Placeholder_1Placeholder*
shape:*
_output_shapes
:*
dtype0
R
Placeholder_2Placeholder*
_output_shapes
:*
shape:*
dtype0

*Embedding/Initializer/random_uniform/shapeConst*
_class
loc:@Embedding*
_output_shapes
:*
dtype0*
valueB"      

(Embedding/Initializer/random_uniform/minConst*
valueB
 *oGŘ˝*
_class
loc:@Embedding*
dtype0*
_output_shapes
: 

(Embedding/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *oGŘ=*
_class
loc:@Embedding
Ý
2Embedding/Initializer/random_uniform/RandomUniformRandomUniform*Embedding/Initializer/random_uniform/shape*

seed *
_class
loc:@Embedding*
_output_shapes
:	*
T0*
dtype0*
seed2 
Â
(Embedding/Initializer/random_uniform/subSub(Embedding/Initializer/random_uniform/max(Embedding/Initializer/random_uniform/min*
_class
loc:@Embedding*
T0*
_output_shapes
: 
Ő
(Embedding/Initializer/random_uniform/mulMul2Embedding/Initializer/random_uniform/RandomUniform(Embedding/Initializer/random_uniform/sub*
_class
loc:@Embedding*
T0*
_output_shapes
:	
Ç
$Embedding/Initializer/random_uniformAdd(Embedding/Initializer/random_uniform/mul(Embedding/Initializer/random_uniform/min*
_class
loc:@Embedding*
T0*
_output_shapes
:	

	Embedding
VariableV2*
shared_name *
dtype0*
	container *
shape:	*
_class
loc:@Embedding*
_output_shapes
:	
ź
Embedding/AssignAssign	Embedding$Embedding/Initializer/random_uniform*
use_locking(*
validate_shape(*
T0*
_class
loc:@Embedding*
_output_shapes
:	
m
Embedding/readIdentity	Embedding*
T0*
_class
loc:@Embedding*
_output_shapes
:	
u
embedding_lookup/axisConst*
_output_shapes
: *
_class
loc:@Embedding*
dtype0*
value	B : 
Ű
embedding_lookupGatherV2Embedding/readPlaceholderembedding_lookup/axis*
Tparams0*

batch_dims *
Taxis0*
_class
loc:@Embedding*
Tindices0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
o
embedding_lookup/IdentityIdentityembedding_lookup*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
]
DropoutWrapperInit/ConstConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
_
DropoutWrapperInit/Const_1Const*
dtype0*
_output_shapes
: *
valueB
 *  ?
_
DropoutWrapperInit/Const_2Const*
dtype0*
valueB
 *   ?*
_output_shapes
: 
_
DropoutWrapperInit_1/ConstConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
a
DropoutWrapperInit_1/Const_1Const*
dtype0*
_output_shapes
: *
valueB
 *  ?
a
DropoutWrapperInit_1/Const_2Const*
dtype0*
valueB
 *   ?*
_output_shapes
: 

4DropoutWrapperZeroState/BasicLSTMCellZeroState/ConstConst*
valueB:*
dtype0*
_output_shapes
:

6DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_1Const*
dtype0*
valueB:@*
_output_shapes
:
|
:DropoutWrapperZeroState/BasicLSTMCellZeroState/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
Ľ
5DropoutWrapperZeroState/BasicLSTMCellZeroState/concatConcatV24DropoutWrapperZeroState/BasicLSTMCellZeroState/Const6DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_1:DropoutWrapperZeroState/BasicLSTMCellZeroState/concat/axis*

Tidx0*
_output_shapes
:*
N*
T0

:DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ë
4DropoutWrapperZeroState/BasicLSTMCellZeroState/zerosFill5DropoutWrapperZeroState/BasicLSTMCellZeroState/concat:DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros/Const*

index_type0*
T0*
_output_shapes
:	@

6DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_2Const*
_output_shapes
:*
dtype0*
valueB:

6DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_3Const*
valueB:@*
_output_shapes
:*
dtype0

6DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_4Const*
_output_shapes
:*
dtype0*
valueB:

6DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_5Const*
dtype0*
valueB:@*
_output_shapes
:
~
<DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ť
7DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1ConcatV26DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_46DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_5<DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1/axis*
_output_shapes
:*
T0*
N*

Tidx0

<DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
ń
6DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1Fill7DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1<DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1/Const*
T0*
_output_shapes
:	@*

index_type0

6DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_6Const*
dtype0*
valueB:*
_output_shapes
:

6DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_7Const*
_output_shapes
:*
valueB:@*
dtype0

6DropoutWrapperZeroState_1/BasicLSTMCellZeroState/ConstConst*
_output_shapes
:*
valueB:*
dtype0

8DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_1Const*
_output_shapes
:*
valueB:@*
dtype0
~
<DropoutWrapperZeroState_1/BasicLSTMCellZeroState/concat/axisConst*
_output_shapes
: *
value	B : *
dtype0
­
7DropoutWrapperZeroState_1/BasicLSTMCellZeroState/concatConcatV26DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const8DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_1<DropoutWrapperZeroState_1/BasicLSTMCellZeroState/concat/axis*

Tidx0*
T0*
_output_shapes
:*
N

<DropoutWrapperZeroState_1/BasicLSTMCellZeroState/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
ń
6DropoutWrapperZeroState_1/BasicLSTMCellZeroState/zerosFill7DropoutWrapperZeroState_1/BasicLSTMCellZeroState/concat<DropoutWrapperZeroState_1/BasicLSTMCellZeroState/zeros/Const*

index_type0*
T0*
_output_shapes
:	@

8DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_2Const*
valueB:*
_output_shapes
:*
dtype0

8DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_3Const*
dtype0*
_output_shapes
:*
valueB:@

8DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_4Const*
valueB:*
_output_shapes
:*
dtype0

8DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_5Const*
valueB:@*
dtype0*
_output_shapes
:

>DropoutWrapperZeroState_1/BasicLSTMCellZeroState/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
ł
9DropoutWrapperZeroState_1/BasicLSTMCellZeroState/concat_1ConcatV28DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_48DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_5>DropoutWrapperZeroState_1/BasicLSTMCellZeroState/concat_1/axis*
_output_shapes
:*
T0*
N*

Tidx0

>DropoutWrapperZeroState_1/BasicLSTMCellZeroState/zeros_1/ConstConst*
dtype0*
valueB
 *    *
_output_shapes
: 
÷
8DropoutWrapperZeroState_1/BasicLSTMCellZeroState/zeros_1Fill9DropoutWrapperZeroState_1/BasicLSTMCellZeroState/concat_1>DropoutWrapperZeroState_1/BasicLSTMCellZeroState/zeros_1/Const*
T0*
_output_shapes
:	@*

index_type0

8DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

8DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_7Const*
_output_shapes
:*
dtype0*
valueB:@
^
bidirectional_rnn/fw/fw/RankConst*
dtype0*
value	B :*
_output_shapes
: 
e
#bidirectional_rnn/fw/fw/range/startConst*
_output_shapes
: *
dtype0*
value	B :
e
#bidirectional_rnn/fw/fw/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
ś
bidirectional_rnn/fw/fw/rangeRange#bidirectional_rnn/fw/fw/range/startbidirectional_rnn/fw/fw/Rank#bidirectional_rnn/fw/fw/range/delta*

Tidx0*
_output_shapes
:
x
'bidirectional_rnn/fw/fw/concat/values_0Const*
_output_shapes
:*
valueB"       *
dtype0
e
#bidirectional_rnn/fw/fw/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
Ń
bidirectional_rnn/fw/fw/concatConcatV2'bidirectional_rnn/fw/fw/concat/values_0bidirectional_rnn/fw/fw/range#bidirectional_rnn/fw/fw/concat/axis*
T0*
_output_shapes
:*

Tidx0*
N
Ž
!bidirectional_rnn/fw/fw/transpose	Transposeembedding_lookup/Identitybidirectional_rnn/fw/fw/concat*
Tperm0*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
e
'bidirectional_rnn/fw/fw/sequence_lengthIdentityPlaceholder_2*
T0*
_output_shapes
:

bidirectional_rnn/fw/fw/ShapeShape'bidirectional_rnn/fw/fw/sequence_length*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
out_type0
h
bidirectional_rnn/fw/fw/stackConst*
_output_shapes
:*
valueB:*
dtype0

bidirectional_rnn/fw/fw/EqualEqualbidirectional_rnn/fw/fw/Shapebidirectional_rnn/fw/fw/stack*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
g
bidirectional_rnn/fw/fw/ConstConst*
valueB: *
_output_shapes
:*
dtype0

bidirectional_rnn/fw/fw/AllAllbidirectional_rnn/fw/fw/Equalbidirectional_rnn/fw/fw/Const*
_output_shapes
: *

Tidx0*
	keep_dims( 
Ź
$bidirectional_rnn/fw/fw/Assert/ConstConst*
dtype0*X
valueOBM BGExpected shape for Tensor bidirectional_rnn/fw/fw/sequence_length:0 is *
_output_shapes
: 
w
&bidirectional_rnn/fw/fw/Assert/Const_1Const*!
valueB B but saw shape: *
_output_shapes
: *
dtype0
´
,bidirectional_rnn/fw/fw/Assert/Assert/data_0Const*X
valueOBM BGExpected shape for Tensor bidirectional_rnn/fw/fw/sequence_length:0 is *
dtype0*
_output_shapes
: 
}
,bidirectional_rnn/fw/fw/Assert/Assert/data_2Const*
dtype0*
_output_shapes
: *!
valueB B but saw shape: 

%bidirectional_rnn/fw/fw/Assert/AssertAssertbidirectional_rnn/fw/fw/All,bidirectional_rnn/fw/fw/Assert/Assert/data_0bidirectional_rnn/fw/fw/stack,bidirectional_rnn/fw/fw/Assert/Assert/data_2bidirectional_rnn/fw/fw/Shape*
T
2*
	summarize
Ł
#bidirectional_rnn/fw/fw/CheckSeqLenIdentity'bidirectional_rnn/fw/fw/sequence_length&^bidirectional_rnn/fw/fw/Assert/Assert*
_output_shapes
:*
T0

bidirectional_rnn/fw/fw/Shape_1Shape!bidirectional_rnn/fw/fw/transpose*
T0*
_output_shapes
:*
out_type0
u
+bidirectional_rnn/fw/fw/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
w
-bidirectional_rnn/fw/fw/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
w
-bidirectional_rnn/fw/fw/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
ó
%bidirectional_rnn/fw/fw/strided_sliceStridedSlicebidirectional_rnn/fw/fw/Shape_1+bidirectional_rnn/fw/fw/strided_slice/stack-bidirectional_rnn/fw/fw/strided_slice/stack_1-bidirectional_rnn/fw/fw/strided_slice/stack_2*
end_mask *

begin_mask *
new_axis_mask *
_output_shapes
: *
Index0*
shrink_axis_mask*
ellipsis_mask *
T0
j
bidirectional_rnn/fw/fw/Const_1Const*
dtype0*
_output_shapes
:*
valueB:
i
bidirectional_rnn/fw/fw/Const_2Const*
dtype0*
valueB:@*
_output_shapes
:
g
%bidirectional_rnn/fw/fw/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
Ď
 bidirectional_rnn/fw/fw/concat_1ConcatV2bidirectional_rnn/fw/fw/Const_1bidirectional_rnn/fw/fw/Const_2%bidirectional_rnn/fw/fw/concat_1/axis*
N*

Tidx0*
_output_shapes
:*
T0
h
#bidirectional_rnn/fw/fw/zeros/ConstConst*
valueB
 *    *
_output_shapes
: *
dtype0
¨
bidirectional_rnn/fw/fw/zerosFill bidirectional_rnn/fw/fw/concat_1#bidirectional_rnn/fw/fw/zeros/Const*
_output_shapes
:	@*
T0*

index_type0
l
bidirectional_rnn/fw/fw/Rank_1Rank#bidirectional_rnn/fw/fw/CheckSeqLen*
_output_shapes
: *
T0
g
%bidirectional_rnn/fw/fw/range_1/startConst*
dtype0*
value	B : *
_output_shapes
: 
g
%bidirectional_rnn/fw/fw/range_1/deltaConst*
value	B :*
_output_shapes
: *
dtype0
Ç
bidirectional_rnn/fw/fw/range_1Range%bidirectional_rnn/fw/fw/range_1/startbidirectional_rnn/fw/fw/Rank_1%bidirectional_rnn/fw/fw/range_1/delta*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ś
bidirectional_rnn/fw/fw/MinMin#bidirectional_rnn/fw/fw/CheckSeqLenbidirectional_rnn/fw/fw/range_1*

Tidx0*
_output_shapes
: *
T0*
	keep_dims( 
l
bidirectional_rnn/fw/fw/Rank_2Rank#bidirectional_rnn/fw/fw/CheckSeqLen*
_output_shapes
: *
T0
g
%bidirectional_rnn/fw/fw/range_2/startConst*
_output_shapes
: *
dtype0*
value	B : 
g
%bidirectional_rnn/fw/fw/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :
Ç
bidirectional_rnn/fw/fw/range_2Range%bidirectional_rnn/fw/fw/range_2/startbidirectional_rnn/fw/fw/Rank_2%bidirectional_rnn/fw/fw/range_2/delta*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tidx0
Ś
bidirectional_rnn/fw/fw/MaxMax#bidirectional_rnn/fw/fw/CheckSeqLenbidirectional_rnn/fw/fw/range_2*
_output_shapes
: *

Tidx0*
T0*
	keep_dims( 
^
bidirectional_rnn/fw/fw/timeConst*
_output_shapes
: *
value	B : *
dtype0
ľ
#bidirectional_rnn/fw/fw/TensorArrayTensorArrayV3%bidirectional_rnn/fw/fw/strided_slice*C
tensor_array_name.,bidirectional_rnn/fw/fw/dynamic_rnn/output_0*
_output_shapes

:: *
identical_element_shapes(*
clear_after_read(*
element_shape:	@*
dtype0*
dynamic_size( 
ˇ
%bidirectional_rnn/fw/fw/TensorArray_1TensorArrayV3%bidirectional_rnn/fw/fw/strided_slice*B
tensor_array_name-+bidirectional_rnn/fw/fw/dynamic_rnn/input_0*
dynamic_size( *
element_shape:
*
identical_element_shapes(*
clear_after_read(*
dtype0*
_output_shapes

:: 

0bidirectional_rnn/fw/fw/TensorArrayUnstack/ShapeShape!bidirectional_rnn/fw/fw/transpose*
_output_shapes
:*
out_type0*
T0

>bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 

@bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:

@bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
Đ
8bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_sliceStridedSlice0bidirectional_rnn/fw/fw/TensorArrayUnstack/Shape>bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stack@bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stack_1@bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stack_2*
end_mask *
ellipsis_mask *
_output_shapes
: *
T0*
new_axis_mask *
Index0*
shrink_axis_mask*

begin_mask 
x
6bidirectional_rnn/fw/fw/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
x
6bidirectional_rnn/fw/fw/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 

0bidirectional_rnn/fw/fw/TensorArrayUnstack/rangeRange6bidirectional_rnn/fw/fw/TensorArrayUnstack/range/start8bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice6bidirectional_rnn/fw/fw/TensorArrayUnstack/range/delta*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ć
Rbidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3%bidirectional_rnn/fw/fw/TensorArray_10bidirectional_rnn/fw/fw/TensorArrayUnstack/range!bidirectional_rnn/fw/fw/transpose'bidirectional_rnn/fw/fw/TensorArray_1:1*
T0*4
_class*
(&loc:@bidirectional_rnn/fw/fw/transpose*
_output_shapes
: 
c
!bidirectional_rnn/fw/fw/Maximum/xConst*
_output_shapes
: *
value	B :*
dtype0

bidirectional_rnn/fw/fw/MaximumMaximum!bidirectional_rnn/fw/fw/Maximum/xbidirectional_rnn/fw/fw/Max*
_output_shapes
: *
T0

bidirectional_rnn/fw/fw/MinimumMinimum%bidirectional_rnn/fw/fw/strided_slicebidirectional_rnn/fw/fw/Maximum*
T0*
_output_shapes
: 
q
/bidirectional_rnn/fw/fw/while/iteration_counterConst*
dtype0*
value	B : *
_output_shapes
: 
é
#bidirectional_rnn/fw/fw/while/EnterEnter/bidirectional_rnn/fw/fw/while/iteration_counter*
_output_shapes
: *;

frame_name-+bidirectional_rnn/fw/fw/while/while_context*
T0*
is_constant( *
parallel_iterations 
Ř
%bidirectional_rnn/fw/fw/while/Enter_1Enterbidirectional_rnn/fw/fw/time*
is_constant( *;

frame_name-+bidirectional_rnn/fw/fw/while/while_context*
parallel_iterations *
T0*
_output_shapes
: 
á
%bidirectional_rnn/fw/fw/while/Enter_2Enter%bidirectional_rnn/fw/fw/TensorArray:1*
T0*
is_constant( *;

frame_name-+bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
: *
parallel_iterations 
ů
%bidirectional_rnn/fw/fw/while/Enter_3Enter4DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context*
T0*
parallel_iterations *
is_constant( *
_output_shapes
:	@
ű
%bidirectional_rnn/fw/fw/while/Enter_4Enter6DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context*
T0*
is_constant( *
parallel_iterations *
_output_shapes
:	@
Ş
#bidirectional_rnn/fw/fw/while/MergeMerge#bidirectional_rnn/fw/fw/while/Enter+bidirectional_rnn/fw/fw/while/NextIteration*
N*
_output_shapes
: : *
T0
°
%bidirectional_rnn/fw/fw/while/Merge_1Merge%bidirectional_rnn/fw/fw/while/Enter_1-bidirectional_rnn/fw/fw/while/NextIteration_1*
_output_shapes
: : *
N*
T0
°
%bidirectional_rnn/fw/fw/while/Merge_2Merge%bidirectional_rnn/fw/fw/while/Enter_2-bidirectional_rnn/fw/fw/while/NextIteration_2*
T0*
N*
_output_shapes
: : 
š
%bidirectional_rnn/fw/fw/while/Merge_3Merge%bidirectional_rnn/fw/fw/while/Enter_3-bidirectional_rnn/fw/fw/while/NextIteration_3*!
_output_shapes
:	@: *
N*
T0
š
%bidirectional_rnn/fw/fw/while/Merge_4Merge%bidirectional_rnn/fw/fw/while/Enter_4-bidirectional_rnn/fw/fw/while/NextIteration_4*
N*
T0*!
_output_shapes
:	@: 

"bidirectional_rnn/fw/fw/while/LessLess#bidirectional_rnn/fw/fw/while/Merge(bidirectional_rnn/fw/fw/while/Less/Enter*
_output_shapes
: *
T0
ä
(bidirectional_rnn/fw/fw/while/Less/EnterEnter%bidirectional_rnn/fw/fw/strided_slice*
_output_shapes
: *;

frame_name-+bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
T0*
parallel_iterations 
 
$bidirectional_rnn/fw/fw/while/Less_1Less%bidirectional_rnn/fw/fw/while/Merge_1*bidirectional_rnn/fw/fw/while/Less_1/Enter*
_output_shapes
: *
T0
ŕ
*bidirectional_rnn/fw/fw/while/Less_1/EnterEnterbidirectional_rnn/fw/fw/Minimum*
_output_shapes
: *
T0*
is_constant(*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context*
parallel_iterations 

(bidirectional_rnn/fw/fw/while/LogicalAnd
LogicalAnd"bidirectional_rnn/fw/fw/while/Less$bidirectional_rnn/fw/fw/while/Less_1*
_output_shapes
: 
t
&bidirectional_rnn/fw/fw/while/LoopCondLoopCond(bidirectional_rnn/fw/fw/while/LogicalAnd*
_output_shapes
: 
Ö
$bidirectional_rnn/fw/fw/while/SwitchSwitch#bidirectional_rnn/fw/fw/while/Merge&bidirectional_rnn/fw/fw/while/LoopCond*6
_class,
*(loc:@bidirectional_rnn/fw/fw/while/Merge*
_output_shapes
: : *
T0
Ü
&bidirectional_rnn/fw/fw/while/Switch_1Switch%bidirectional_rnn/fw/fw/while/Merge_1&bidirectional_rnn/fw/fw/while/LoopCond*8
_class.
,*loc:@bidirectional_rnn/fw/fw/while/Merge_1*
T0*
_output_shapes
: : 
Ü
&bidirectional_rnn/fw/fw/while/Switch_2Switch%bidirectional_rnn/fw/fw/while/Merge_2&bidirectional_rnn/fw/fw/while/LoopCond*
T0*8
_class.
,*loc:@bidirectional_rnn/fw/fw/while/Merge_2*
_output_shapes
: : 
î
&bidirectional_rnn/fw/fw/while/Switch_3Switch%bidirectional_rnn/fw/fw/while/Merge_3&bidirectional_rnn/fw/fw/while/LoopCond*8
_class.
,*loc:@bidirectional_rnn/fw/fw/while/Merge_3*
T0**
_output_shapes
:	@:	@
î
&bidirectional_rnn/fw/fw/while/Switch_4Switch%bidirectional_rnn/fw/fw/while/Merge_4&bidirectional_rnn/fw/fw/while/LoopCond*8
_class.
,*loc:@bidirectional_rnn/fw/fw/while/Merge_4*
T0**
_output_shapes
:	@:	@
{
&bidirectional_rnn/fw/fw/while/IdentityIdentity&bidirectional_rnn/fw/fw/while/Switch:1*
T0*
_output_shapes
: 

(bidirectional_rnn/fw/fw/while/Identity_1Identity(bidirectional_rnn/fw/fw/while/Switch_1:1*
_output_shapes
: *
T0

(bidirectional_rnn/fw/fw/while/Identity_2Identity(bidirectional_rnn/fw/fw/while/Switch_2:1*
_output_shapes
: *
T0

(bidirectional_rnn/fw/fw/while/Identity_3Identity(bidirectional_rnn/fw/fw/while/Switch_3:1*
T0*
_output_shapes
:	@

(bidirectional_rnn/fw/fw/while/Identity_4Identity(bidirectional_rnn/fw/fw/while/Switch_4:1*
_output_shapes
:	@*
T0

#bidirectional_rnn/fw/fw/while/add/yConst'^bidirectional_rnn/fw/fw/while/Identity*
_output_shapes
: *
dtype0*
value	B :

!bidirectional_rnn/fw/fw/while/addAdd&bidirectional_rnn/fw/fw/while/Identity#bidirectional_rnn/fw/fw/while/add/y*
_output_shapes
: *
T0

/bidirectional_rnn/fw/fw/while/TensorArrayReadV3TensorArrayReadV35bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter(bidirectional_rnn/fw/fw/while/Identity_17bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1* 
_output_shapes
:
*
dtype0
ő
5bidirectional_rnn/fw/fw/while/TensorArrayReadV3/EnterEnter%bidirectional_rnn/fw/fw/TensorArray_1*
_output_shapes
:*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context*
parallel_iterations *
T0*
is_constant(
 
7bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1EnterRbidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
_output_shapes
: *
T0*
is_constant(*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context*
parallel_iterations 
š
*bidirectional_rnn/fw/fw/while/GreaterEqualGreaterEqual(bidirectional_rnn/fw/fw/while/Identity_10bidirectional_rnn/fw/fw/while/GreaterEqual/Enter*
T0*
_output_shapes
:
ě
0bidirectional_rnn/fw/fw/while/GreaterEqual/EnterEnter#bidirectional_rnn/fw/fw/CheckSeqLen*
_output_shapes
:*
is_constant(*
T0*
parallel_iterations *;

frame_name-+bidirectional_rnn/fw/fw/while/while_context
Ý
Lbidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*>
_class4
20loc:@bidirectional_rnn/fw/basic_lstm_cell/kernel*
dtype0*
valueB"@     
Ď
Jbidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/minConst*
dtype0*
valueB
 *ňę­˝*>
_class4
20loc:@bidirectional_rnn/fw/basic_lstm_cell/kernel*
_output_shapes
: 
Ď
Jbidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *ňę­=*>
_class4
20loc:@bidirectional_rnn/fw/basic_lstm_cell/kernel
Ä
Tbidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformLbidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/shape*
T0*

seed *>
_class4
20loc:@bidirectional_rnn/fw/basic_lstm_cell/kernel* 
_output_shapes
:
Ŕ*
seed2 *
dtype0
Ę
Jbidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/subSubJbidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/maxJbidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@bidirectional_rnn/fw/basic_lstm_cell/kernel*
_output_shapes
: 
Ţ
Jbidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/mulMulTbidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformJbidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
Ŕ*
T0*>
_class4
20loc:@bidirectional_rnn/fw/basic_lstm_cell/kernel
Đ
Fbidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniformAddJbidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/mulJbidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/min* 
_output_shapes
:
Ŕ*>
_class4
20loc:@bidirectional_rnn/fw/basic_lstm_cell/kernel*
T0
ă
+bidirectional_rnn/fw/basic_lstm_cell/kernel
VariableV2*
shape:
Ŕ*
dtype0*
	container *>
_class4
20loc:@bidirectional_rnn/fw/basic_lstm_cell/kernel* 
_output_shapes
:
Ŕ*
shared_name 
Ĺ
2bidirectional_rnn/fw/basic_lstm_cell/kernel/AssignAssign+bidirectional_rnn/fw/basic_lstm_cell/kernelFbidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform*>
_class4
20loc:@bidirectional_rnn/fw/basic_lstm_cell/kernel*
T0*
validate_shape(* 
_output_shapes
:
Ŕ*
use_locking(

0bidirectional_rnn/fw/basic_lstm_cell/kernel/readIdentity+bidirectional_rnn/fw/basic_lstm_cell/kernel*
T0* 
_output_shapes
:
Ŕ
Č
;bidirectional_rnn/fw/basic_lstm_cell/bias/Initializer/zerosConst*
valueB*    *<
_class2
0.loc:@bidirectional_rnn/fw/basic_lstm_cell/bias*
dtype0*
_output_shapes	
:
Ő
)bidirectional_rnn/fw/basic_lstm_cell/bias
VariableV2*
_output_shapes	
:*<
_class2
0.loc:@bidirectional_rnn/fw/basic_lstm_cell/bias*
	container *
shared_name *
dtype0*
shape:
Ż
0bidirectional_rnn/fw/basic_lstm_cell/bias/AssignAssign)bidirectional_rnn/fw/basic_lstm_cell/bias;bidirectional_rnn/fw/basic_lstm_cell/bias/Initializer/zeros*<
_class2
0.loc:@bidirectional_rnn/fw/basic_lstm_cell/bias*
T0*
validate_shape(*
_output_shapes	
:*
use_locking(

.bidirectional_rnn/fw/basic_lstm_cell/bias/readIdentity)bidirectional_rnn/fw/basic_lstm_cell/bias*
_output_shapes	
:*
T0

3bidirectional_rnn/fw/fw/while/basic_lstm_cell/ConstConst'^bidirectional_rnn/fw/fw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
¤
9bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat/axisConst'^bidirectional_rnn/fw/fw/while/Identity*
dtype0*
_output_shapes
: *
value	B :

4bidirectional_rnn/fw/fw/while/basic_lstm_cell/concatConcatV2/bidirectional_rnn/fw/fw/while/TensorArrayReadV3(bidirectional_rnn/fw/fw/while/Identity_49bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat/axis*
N* 
_output_shapes
:
Ŕ*

Tidx0*
T0

4bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMulMatMul4bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat:bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul/Enter* 
_output_shapes
:
*
transpose_b( *
transpose_a( *
T0

:bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul/EnterEnter0bidirectional_rnn/fw/basic_lstm_cell/kernel/read*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context*
parallel_iterations *
is_constant(* 
_output_shapes
:
Ŕ*
T0
ő
5bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAddBiasAdd4bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul;bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd/Enter*
T0* 
_output_shapes
:
*
data_formatNHWC

;bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd/EnterEnter.bidirectional_rnn/fw/basic_lstm_cell/bias/read*
T0*
parallel_iterations *;

frame_name-+bidirectional_rnn/fw/fw/while/while_context*
_output_shapes	
:*
is_constant(
 
5bidirectional_rnn/fw/fw/while/basic_lstm_cell/Const_1Const'^bidirectional_rnn/fw/fw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 

3bidirectional_rnn/fw/fw/while/basic_lstm_cell/splitSplit3bidirectional_rnn/fw/fw/while/basic_lstm_cell/Const5bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd*
	num_split*
T0*@
_output_shapes.
,:	@:	@:	@:	@
Ł
5bidirectional_rnn/fw/fw/while/basic_lstm_cell/Const_2Const'^bidirectional_rnn/fw/fw/while/Identity*
_output_shapes
: *
valueB
 *    *
dtype0
Đ
1bidirectional_rnn/fw/fw/while/basic_lstm_cell/AddAdd5bidirectional_rnn/fw/fw/while/basic_lstm_cell/split:25bidirectional_rnn/fw/fw/while/basic_lstm_cell/Const_2*
T0*
_output_shapes
:	@

5bidirectional_rnn/fw/fw/while/basic_lstm_cell/SigmoidSigmoid1bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add*
_output_shapes
:	@*
T0
Ă
1bidirectional_rnn/fw/fw/while/basic_lstm_cell/MulMul(bidirectional_rnn/fw/fw/while/Identity_35bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid*
T0*
_output_shapes
:	@
Ą
7bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid_1Sigmoid3bidirectional_rnn/fw/fw/while/basic_lstm_cell/split*
_output_shapes
:	@*
T0

2bidirectional_rnn/fw/fw/while/basic_lstm_cell/TanhTanh5bidirectional_rnn/fw/fw/while/basic_lstm_cell/split:1*
_output_shapes
:	@*
T0
Ń
3bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1Mul7bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid_12bidirectional_rnn/fw/fw/while/basic_lstm_cell/Tanh*
T0*
_output_shapes
:	@
Ě
3bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_1Add1bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul3bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1*
T0*
_output_shapes
:	@

4bidirectional_rnn/fw/fw/while/basic_lstm_cell/Tanh_1Tanh3bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_1*
_output_shapes
:	@*
T0
Ł
7bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid_2Sigmoid5bidirectional_rnn/fw/fw/while/basic_lstm_cell/split:3*
_output_shapes
:	@*
T0
Ó
3bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2Mul4bidirectional_rnn/fw/fw/while/basic_lstm_cell/Tanh_17bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid_2*
T0*
_output_shapes
:	@

*bidirectional_rnn/fw/fw/while/dropout/rateConst'^bidirectional_rnn/fw/fw/while/Identity*
valueB
 *   ?*
_output_shapes
: *
dtype0
Ľ
+bidirectional_rnn/fw/fw/while/dropout/ShapeConst'^bidirectional_rnn/fw/fw/while/Identity*
_output_shapes
:*
dtype0*
valueB"   @   
Ś
8bidirectional_rnn/fw/fw/while/dropout/random_uniform/minConst'^bidirectional_rnn/fw/fw/while/Identity*
_output_shapes
: *
valueB
 *    *
dtype0
Ś
8bidirectional_rnn/fw/fw/while/dropout/random_uniform/maxConst'^bidirectional_rnn/fw/fw/while/Identity*
dtype0*
valueB
 *  ?*
_output_shapes
: 
Đ
Bbidirectional_rnn/fw/fw/while/dropout/random_uniform/RandomUniformRandomUniform+bidirectional_rnn/fw/fw/while/dropout/Shape*
dtype0*
T0*

seed *
_output_shapes
:	@*
seed2 
Ô
8bidirectional_rnn/fw/fw/while/dropout/random_uniform/subSub8bidirectional_rnn/fw/fw/while/dropout/random_uniform/max8bidirectional_rnn/fw/fw/while/dropout/random_uniform/min*
T0*
_output_shapes
: 
ç
8bidirectional_rnn/fw/fw/while/dropout/random_uniform/mulMulBbidirectional_rnn/fw/fw/while/dropout/random_uniform/RandomUniform8bidirectional_rnn/fw/fw/while/dropout/random_uniform/sub*
_output_shapes
:	@*
T0
Ů
4bidirectional_rnn/fw/fw/while/dropout/random_uniformAdd8bidirectional_rnn/fw/fw/while/dropout/random_uniform/mul8bidirectional_rnn/fw/fw/while/dropout/random_uniform/min*
_output_shapes
:	@*
T0

+bidirectional_rnn/fw/fw/while/dropout/sub/xConst'^bidirectional_rnn/fw/fw/while/Identity*
_output_shapes
: *
valueB
 *  ?*
dtype0
Ş
)bidirectional_rnn/fw/fw/while/dropout/subSub+bidirectional_rnn/fw/fw/while/dropout/sub/x*bidirectional_rnn/fw/fw/while/dropout/rate*
_output_shapes
: *
T0

/bidirectional_rnn/fw/fw/while/dropout/truediv/xConst'^bidirectional_rnn/fw/fw/while/Identity*
valueB
 *  ?*
dtype0*
_output_shapes
: 
ľ
-bidirectional_rnn/fw/fw/while/dropout/truedivRealDiv/bidirectional_rnn/fw/fw/while/dropout/truediv/x)bidirectional_rnn/fw/fw/while/dropout/sub*
T0*
_output_shapes
: 
Î
2bidirectional_rnn/fw/fw/while/dropout/GreaterEqualGreaterEqual4bidirectional_rnn/fw/fw/while/dropout/random_uniform*bidirectional_rnn/fw/fw/while/dropout/rate*
_output_shapes
:	@*
T0
ž
)bidirectional_rnn/fw/fw/while/dropout/mulMul3bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2-bidirectional_rnn/fw/fw/while/dropout/truediv*
T0*
_output_shapes
:	@
Ż
*bidirectional_rnn/fw/fw/while/dropout/CastCast2bidirectional_rnn/fw/fw/while/dropout/GreaterEqual*
Truncate( *

SrcT0
*

DstT0*
_output_shapes
:	@
ł
+bidirectional_rnn/fw/fw/while/dropout/mul_1Mul)bidirectional_rnn/fw/fw/while/dropout/mul*bidirectional_rnn/fw/fw/while/dropout/Cast*
T0*
_output_shapes
:	@

$bidirectional_rnn/fw/fw/while/SelectSelect*bidirectional_rnn/fw/fw/while/GreaterEqual*bidirectional_rnn/fw/fw/while/Select/Enter+bidirectional_rnn/fw/fw/while/dropout/mul_1*>
_class4
20loc:@bidirectional_rnn/fw/fw/while/dropout/mul_1*
_output_shapes
:	@*
T0
§
*bidirectional_rnn/fw/fw/while/Select/EnterEnterbidirectional_rnn/fw/fw/zeros*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context*>
_class4
20loc:@bidirectional_rnn/fw/fw/while/dropout/mul_1*
parallel_iterations *
_output_shapes
:	@*
T0*
is_constant(
­
&bidirectional_rnn/fw/fw/while/Select_1Select*bidirectional_rnn/fw/fw/while/GreaterEqual(bidirectional_rnn/fw/fw/while/Identity_33bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_1*F
_class<
:8loc:@bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_1*
T0*
_output_shapes
:	@
­
&bidirectional_rnn/fw/fw/while/Select_2Select*bidirectional_rnn/fw/fw/while/GreaterEqual(bidirectional_rnn/fw/fw/while/Identity_43bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2*
_output_shapes
:	@*F
_class<
:8loc:@bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2*
T0
ű
Abidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Gbidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3/Enter(bidirectional_rnn/fw/fw/while/Identity_1$bidirectional_rnn/fw/fw/while/Select(bidirectional_rnn/fw/fw/while/Identity_2*>
_class4
20loc:@bidirectional_rnn/fw/fw/while/dropout/mul_1*
_output_shapes
: *
T0
Ĺ
Gbidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnter#bidirectional_rnn/fw/fw/TensorArray*
T0*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context*>
_class4
20loc:@bidirectional_rnn/fw/fw/while/dropout/mul_1*
parallel_iterations *
_output_shapes
:*
is_constant(

%bidirectional_rnn/fw/fw/while/add_1/yConst'^bidirectional_rnn/fw/fw/while/Identity*
_output_shapes
: *
dtype0*
value	B :

#bidirectional_rnn/fw/fw/while/add_1Add(bidirectional_rnn/fw/fw/while/Identity_1%bidirectional_rnn/fw/fw/while/add_1/y*
_output_shapes
: *
T0

+bidirectional_rnn/fw/fw/while/NextIterationNextIteration!bidirectional_rnn/fw/fw/while/add*
_output_shapes
: *
T0

-bidirectional_rnn/fw/fw/while/NextIteration_1NextIteration#bidirectional_rnn/fw/fw/while/add_1*
_output_shapes
: *
T0
˘
-bidirectional_rnn/fw/fw/while/NextIteration_2NextIterationAbidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 

-bidirectional_rnn/fw/fw/while/NextIteration_3NextIteration&bidirectional_rnn/fw/fw/while/Select_1*
_output_shapes
:	@*
T0

-bidirectional_rnn/fw/fw/while/NextIteration_4NextIteration&bidirectional_rnn/fw/fw/while/Select_2*
_output_shapes
:	@*
T0
q
"bidirectional_rnn/fw/fw/while/ExitExit$bidirectional_rnn/fw/fw/while/Switch*
T0*
_output_shapes
: 
u
$bidirectional_rnn/fw/fw/while/Exit_1Exit&bidirectional_rnn/fw/fw/while/Switch_1*
T0*
_output_shapes
: 
u
$bidirectional_rnn/fw/fw/while/Exit_2Exit&bidirectional_rnn/fw/fw/while/Switch_2*
_output_shapes
: *
T0
~
$bidirectional_rnn/fw/fw/while/Exit_3Exit&bidirectional_rnn/fw/fw/while/Switch_3*
_output_shapes
:	@*
T0
~
$bidirectional_rnn/fw/fw/while/Exit_4Exit&bidirectional_rnn/fw/fw/while/Switch_4*
T0*
_output_shapes
:	@
ę
:bidirectional_rnn/fw/fw/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3#bidirectional_rnn/fw/fw/TensorArray$bidirectional_rnn/fw/fw/while/Exit_2*
_output_shapes
: *6
_class,
*(loc:@bidirectional_rnn/fw/fw/TensorArray
Ž
4bidirectional_rnn/fw/fw/TensorArrayStack/range/startConst*
_output_shapes
: *
dtype0*6
_class,
*(loc:@bidirectional_rnn/fw/fw/TensorArray*
value	B : 
Ž
4bidirectional_rnn/fw/fw/TensorArrayStack/range/deltaConst*6
_class,
*(loc:@bidirectional_rnn/fw/fw/TensorArray*
value	B :*
dtype0*
_output_shapes
: 
Č
.bidirectional_rnn/fw/fw/TensorArrayStack/rangeRange4bidirectional_rnn/fw/fw/TensorArrayStack/range/start:bidirectional_rnn/fw/fw/TensorArrayStack/TensorArraySizeV34bidirectional_rnn/fw/fw/TensorArrayStack/range/delta*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tidx0*6
_class,
*(loc:@bidirectional_rnn/fw/fw/TensorArray
ß
<bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3#bidirectional_rnn/fw/fw/TensorArray.bidirectional_rnn/fw/fw/TensorArrayStack/range$bidirectional_rnn/fw/fw/while/Exit_2*
dtype0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@*6
_class,
*(loc:@bidirectional_rnn/fw/fw/TensorArray*
element_shape:	@
i
bidirectional_rnn/fw/fw/Const_3Const*
dtype0*
valueB:@*
_output_shapes
:
`
bidirectional_rnn/fw/fw/Rank_3Const*
dtype0*
_output_shapes
: *
value	B :
g
%bidirectional_rnn/fw/fw/range_3/startConst*
value	B :*
dtype0*
_output_shapes
: 
g
%bidirectional_rnn/fw/fw/range_3/deltaConst*
_output_shapes
: *
value	B :*
dtype0
ž
bidirectional_rnn/fw/fw/range_3Range%bidirectional_rnn/fw/fw/range_3/startbidirectional_rnn/fw/fw/Rank_3%bidirectional_rnn/fw/fw/range_3/delta*

Tidx0*
_output_shapes
:
z
)bidirectional_rnn/fw/fw/concat_2/values_0Const*
dtype0*
_output_shapes
:*
valueB"       
g
%bidirectional_rnn/fw/fw/concat_2/axisConst*
dtype0*
_output_shapes
: *
value	B : 
Ů
 bidirectional_rnn/fw/fw/concat_2ConcatV2)bidirectional_rnn/fw/fw/concat_2/values_0bidirectional_rnn/fw/fw/range_3%bidirectional_rnn/fw/fw/concat_2/axis*
N*
_output_shapes
:*

Tidx0*
T0
Ô
#bidirectional_rnn/fw/fw/transpose_1	Transpose<bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3 bidirectional_rnn/fw/fw/concat_2*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
Tperm0*
T0
Ĺ
$bidirectional_rnn/bw/ReverseSequenceReverseSequenceembedding_lookup/IdentityPlaceholder_2*

Tlen0*
T0*
seq_dim*
	batch_dim *-
_output_shapes
:˙˙˙˙˙˙˙˙˙
^
bidirectional_rnn/bw/bw/RankConst*
dtype0*
value	B :*
_output_shapes
: 
e
#bidirectional_rnn/bw/bw/range/startConst*
_output_shapes
: *
dtype0*
value	B :
e
#bidirectional_rnn/bw/bw/range/deltaConst*
value	B :*
_output_shapes
: *
dtype0
ś
bidirectional_rnn/bw/bw/rangeRange#bidirectional_rnn/bw/bw/range/startbidirectional_rnn/bw/bw/Rank#bidirectional_rnn/bw/bw/range/delta*
_output_shapes
:*

Tidx0
x
'bidirectional_rnn/bw/bw/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       
e
#bidirectional_rnn/bw/bw/concat/axisConst*
value	B : *
_output_shapes
: *
dtype0
Ń
bidirectional_rnn/bw/bw/concatConcatV2'bidirectional_rnn/bw/bw/concat/values_0bidirectional_rnn/bw/bw/range#bidirectional_rnn/bw/bw/concat/axis*

Tidx0*
N*
_output_shapes
:*
T0
š
!bidirectional_rnn/bw/bw/transpose	Transpose$bidirectional_rnn/bw/ReverseSequencebidirectional_rnn/bw/bw/concat*
T0*
Tperm0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
e
'bidirectional_rnn/bw/bw/sequence_lengthIdentityPlaceholder_2*
_output_shapes
:*
T0

bidirectional_rnn/bw/bw/ShapeShape'bidirectional_rnn/bw/bw/sequence_length*
out_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
h
bidirectional_rnn/bw/bw/stackConst*
valueB:*
dtype0*
_output_shapes
:

bidirectional_rnn/bw/bw/EqualEqualbidirectional_rnn/bw/bw/Shapebidirectional_rnn/bw/bw/stack*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
g
bidirectional_rnn/bw/bw/ConstConst*
_output_shapes
:*
valueB: *
dtype0

bidirectional_rnn/bw/bw/AllAllbidirectional_rnn/bw/bw/Equalbidirectional_rnn/bw/bw/Const*
	keep_dims( *
_output_shapes
: *

Tidx0
Ź
$bidirectional_rnn/bw/bw/Assert/ConstConst*
_output_shapes
: *
dtype0*X
valueOBM BGExpected shape for Tensor bidirectional_rnn/bw/bw/sequence_length:0 is 
w
&bidirectional_rnn/bw/bw/Assert/Const_1Const*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 
´
,bidirectional_rnn/bw/bw/Assert/Assert/data_0Const*X
valueOBM BGExpected shape for Tensor bidirectional_rnn/bw/bw/sequence_length:0 is *
_output_shapes
: *
dtype0
}
,bidirectional_rnn/bw/bw/Assert/Assert/data_2Const*!
valueB B but saw shape: *
_output_shapes
: *
dtype0

%bidirectional_rnn/bw/bw/Assert/AssertAssertbidirectional_rnn/bw/bw/All,bidirectional_rnn/bw/bw/Assert/Assert/data_0bidirectional_rnn/bw/bw/stack,bidirectional_rnn/bw/bw/Assert/Assert/data_2bidirectional_rnn/bw/bw/Shape*
T
2*
	summarize
Ł
#bidirectional_rnn/bw/bw/CheckSeqLenIdentity'bidirectional_rnn/bw/bw/sequence_length&^bidirectional_rnn/bw/bw/Assert/Assert*
T0*
_output_shapes
:

bidirectional_rnn/bw/bw/Shape_1Shape!bidirectional_rnn/bw/bw/transpose*
_output_shapes
:*
out_type0*
T0
u
+bidirectional_rnn/bw/bw/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
w
-bidirectional_rnn/bw/bw/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
w
-bidirectional_rnn/bw/bw/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
ó
%bidirectional_rnn/bw/bw/strided_sliceStridedSlicebidirectional_rnn/bw/bw/Shape_1+bidirectional_rnn/bw/bw/strided_slice/stack-bidirectional_rnn/bw/bw/strided_slice/stack_1-bidirectional_rnn/bw/bw/strided_slice/stack_2*
new_axis_mask *

begin_mask *
shrink_axis_mask*
Index0*
ellipsis_mask *
end_mask *
_output_shapes
: *
T0
j
bidirectional_rnn/bw/bw/Const_1Const*
dtype0*
_output_shapes
:*
valueB:
i
bidirectional_rnn/bw/bw/Const_2Const*
dtype0*
_output_shapes
:*
valueB:@
g
%bidirectional_rnn/bw/bw/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ď
 bidirectional_rnn/bw/bw/concat_1ConcatV2bidirectional_rnn/bw/bw/Const_1bidirectional_rnn/bw/bw/Const_2%bidirectional_rnn/bw/bw/concat_1/axis*
N*
_output_shapes
:*
T0*

Tidx0
h
#bidirectional_rnn/bw/bw/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
¨
bidirectional_rnn/bw/bw/zerosFill bidirectional_rnn/bw/bw/concat_1#bidirectional_rnn/bw/bw/zeros/Const*
T0*
_output_shapes
:	@*

index_type0
l
bidirectional_rnn/bw/bw/Rank_1Rank#bidirectional_rnn/bw/bw/CheckSeqLen*
T0*
_output_shapes
: 
g
%bidirectional_rnn/bw/bw/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 
g
%bidirectional_rnn/bw/bw/range_1/deltaConst*
_output_shapes
: *
value	B :*
dtype0
Ç
bidirectional_rnn/bw/bw/range_1Range%bidirectional_rnn/bw/bw/range_1/startbidirectional_rnn/bw/bw/Rank_1%bidirectional_rnn/bw/bw/range_1/delta*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tidx0
Ś
bidirectional_rnn/bw/bw/MinMin#bidirectional_rnn/bw/bw/CheckSeqLenbidirectional_rnn/bw/bw/range_1*
	keep_dims( *
_output_shapes
: *

Tidx0*
T0
l
bidirectional_rnn/bw/bw/Rank_2Rank#bidirectional_rnn/bw/bw/CheckSeqLen*
_output_shapes
: *
T0
g
%bidirectional_rnn/bw/bw/range_2/startConst*
value	B : *
dtype0*
_output_shapes
: 
g
%bidirectional_rnn/bw/bw/range_2/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
Ç
bidirectional_rnn/bw/bw/range_2Range%bidirectional_rnn/bw/bw/range_2/startbidirectional_rnn/bw/bw/Rank_2%bidirectional_rnn/bw/bw/range_2/delta*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tidx0
Ś
bidirectional_rnn/bw/bw/MaxMax#bidirectional_rnn/bw/bw/CheckSeqLenbidirectional_rnn/bw/bw/range_2*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
^
bidirectional_rnn/bw/bw/timeConst*
dtype0*
_output_shapes
: *
value	B : 
ľ
#bidirectional_rnn/bw/bw/TensorArrayTensorArrayV3%bidirectional_rnn/bw/bw/strided_slice*C
tensor_array_name.,bidirectional_rnn/bw/bw/dynamic_rnn/output_0*
element_shape:	@*
_output_shapes

:: *
dtype0*
identical_element_shapes(*
dynamic_size( *
clear_after_read(
ˇ
%bidirectional_rnn/bw/bw/TensorArray_1TensorArrayV3%bidirectional_rnn/bw/bw/strided_slice*
clear_after_read(*B
tensor_array_name-+bidirectional_rnn/bw/bw/dynamic_rnn/input_0*
identical_element_shapes(*
element_shape:
*
dtype0*
dynamic_size( *
_output_shapes

:: 

0bidirectional_rnn/bw/bw/TensorArrayUnstack/ShapeShape!bidirectional_rnn/bw/bw/transpose*
_output_shapes
:*
T0*
out_type0

>bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
_output_shapes
:*
dtype0

@bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:

@bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
Đ
8bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_sliceStridedSlice0bidirectional_rnn/bw/bw/TensorArrayUnstack/Shape>bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stack@bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stack_1@bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stack_2*
new_axis_mask *

begin_mask *
T0*
_output_shapes
: *
ellipsis_mask *
end_mask *
shrink_axis_mask*
Index0
x
6bidirectional_rnn/bw/bw/TensorArrayUnstack/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
x
6bidirectional_rnn/bw/bw/TensorArrayUnstack/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :

0bidirectional_rnn/bw/bw/TensorArrayUnstack/rangeRange6bidirectional_rnn/bw/bw/TensorArrayUnstack/range/start8bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice6bidirectional_rnn/bw/bw/TensorArrayUnstack/range/delta*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ć
Rbidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3%bidirectional_rnn/bw/bw/TensorArray_10bidirectional_rnn/bw/bw/TensorArrayUnstack/range!bidirectional_rnn/bw/bw/transpose'bidirectional_rnn/bw/bw/TensorArray_1:1*
T0*
_output_shapes
: *4
_class*
(&loc:@bidirectional_rnn/bw/bw/transpose
c
!bidirectional_rnn/bw/bw/Maximum/xConst*
dtype0*
_output_shapes
: *
value	B :

bidirectional_rnn/bw/bw/MaximumMaximum!bidirectional_rnn/bw/bw/Maximum/xbidirectional_rnn/bw/bw/Max*
_output_shapes
: *
T0

bidirectional_rnn/bw/bw/MinimumMinimum%bidirectional_rnn/bw/bw/strided_slicebidirectional_rnn/bw/bw/Maximum*
_output_shapes
: *
T0
q
/bidirectional_rnn/bw/bw/while/iteration_counterConst*
value	B : *
dtype0*
_output_shapes
: 
é
#bidirectional_rnn/bw/bw/while/EnterEnter/bidirectional_rnn/bw/bw/while/iteration_counter*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context*
T0*
parallel_iterations *
is_constant( *
_output_shapes
: 
Ř
%bidirectional_rnn/bw/bw/while/Enter_1Enterbidirectional_rnn/bw/bw/time*
T0*
_output_shapes
: *
parallel_iterations *;

frame_name-+bidirectional_rnn/bw/bw/while/while_context*
is_constant( 
á
%bidirectional_rnn/bw/bw/while/Enter_2Enter%bidirectional_rnn/bw/bw/TensorArray:1*
_output_shapes
: *
is_constant( *
parallel_iterations *;

frame_name-+bidirectional_rnn/bw/bw/while/while_context*
T0
ű
%bidirectional_rnn/bw/bw/while/Enter_3Enter6DropoutWrapperZeroState_1/BasicLSTMCellZeroState/zeros*
is_constant( *
_output_shapes
:	@*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context*
T0*
parallel_iterations 
ý
%bidirectional_rnn/bw/bw/while/Enter_4Enter8DropoutWrapperZeroState_1/BasicLSTMCellZeroState/zeros_1*
parallel_iterations *
T0*
is_constant( *;

frame_name-+bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:	@
Ş
#bidirectional_rnn/bw/bw/while/MergeMerge#bidirectional_rnn/bw/bw/while/Enter+bidirectional_rnn/bw/bw/while/NextIteration*
_output_shapes
: : *
N*
T0
°
%bidirectional_rnn/bw/bw/while/Merge_1Merge%bidirectional_rnn/bw/bw/while/Enter_1-bidirectional_rnn/bw/bw/while/NextIteration_1*
T0*
_output_shapes
: : *
N
°
%bidirectional_rnn/bw/bw/while/Merge_2Merge%bidirectional_rnn/bw/bw/while/Enter_2-bidirectional_rnn/bw/bw/while/NextIteration_2*
_output_shapes
: : *
T0*
N
š
%bidirectional_rnn/bw/bw/while/Merge_3Merge%bidirectional_rnn/bw/bw/while/Enter_3-bidirectional_rnn/bw/bw/while/NextIteration_3*!
_output_shapes
:	@: *
N*
T0
š
%bidirectional_rnn/bw/bw/while/Merge_4Merge%bidirectional_rnn/bw/bw/while/Enter_4-bidirectional_rnn/bw/bw/while/NextIteration_4*
T0*
N*!
_output_shapes
:	@: 

"bidirectional_rnn/bw/bw/while/LessLess#bidirectional_rnn/bw/bw/while/Merge(bidirectional_rnn/bw/bw/while/Less/Enter*
T0*
_output_shapes
: 
ä
(bidirectional_rnn/bw/bw/while/Less/EnterEnter%bidirectional_rnn/bw/bw/strided_slice*
_output_shapes
: *
is_constant(*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context*
T0*
parallel_iterations 
 
$bidirectional_rnn/bw/bw/while/Less_1Less%bidirectional_rnn/bw/bw/while/Merge_1*bidirectional_rnn/bw/bw/while/Less_1/Enter*
T0*
_output_shapes
: 
ŕ
*bidirectional_rnn/bw/bw/while/Less_1/EnterEnterbidirectional_rnn/bw/bw/Minimum*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context*
T0*
_output_shapes
: *
parallel_iterations *
is_constant(

(bidirectional_rnn/bw/bw/while/LogicalAnd
LogicalAnd"bidirectional_rnn/bw/bw/while/Less$bidirectional_rnn/bw/bw/while/Less_1*
_output_shapes
: 
t
&bidirectional_rnn/bw/bw/while/LoopCondLoopCond(bidirectional_rnn/bw/bw/while/LogicalAnd*
_output_shapes
: 
Ö
$bidirectional_rnn/bw/bw/while/SwitchSwitch#bidirectional_rnn/bw/bw/while/Merge&bidirectional_rnn/bw/bw/while/LoopCond*
T0*
_output_shapes
: : *6
_class,
*(loc:@bidirectional_rnn/bw/bw/while/Merge
Ü
&bidirectional_rnn/bw/bw/while/Switch_1Switch%bidirectional_rnn/bw/bw/while/Merge_1&bidirectional_rnn/bw/bw/while/LoopCond*8
_class.
,*loc:@bidirectional_rnn/bw/bw/while/Merge_1*
T0*
_output_shapes
: : 
Ü
&bidirectional_rnn/bw/bw/while/Switch_2Switch%bidirectional_rnn/bw/bw/while/Merge_2&bidirectional_rnn/bw/bw/while/LoopCond*8
_class.
,*loc:@bidirectional_rnn/bw/bw/while/Merge_2*
T0*
_output_shapes
: : 
î
&bidirectional_rnn/bw/bw/while/Switch_3Switch%bidirectional_rnn/bw/bw/while/Merge_3&bidirectional_rnn/bw/bw/while/LoopCond*
T0**
_output_shapes
:	@:	@*8
_class.
,*loc:@bidirectional_rnn/bw/bw/while/Merge_3
î
&bidirectional_rnn/bw/bw/while/Switch_4Switch%bidirectional_rnn/bw/bw/while/Merge_4&bidirectional_rnn/bw/bw/while/LoopCond**
_output_shapes
:	@:	@*8
_class.
,*loc:@bidirectional_rnn/bw/bw/while/Merge_4*
T0
{
&bidirectional_rnn/bw/bw/while/IdentityIdentity&bidirectional_rnn/bw/bw/while/Switch:1*
T0*
_output_shapes
: 

(bidirectional_rnn/bw/bw/while/Identity_1Identity(bidirectional_rnn/bw/bw/while/Switch_1:1*
T0*
_output_shapes
: 

(bidirectional_rnn/bw/bw/while/Identity_2Identity(bidirectional_rnn/bw/bw/while/Switch_2:1*
_output_shapes
: *
T0

(bidirectional_rnn/bw/bw/while/Identity_3Identity(bidirectional_rnn/bw/bw/while/Switch_3:1*
_output_shapes
:	@*
T0

(bidirectional_rnn/bw/bw/while/Identity_4Identity(bidirectional_rnn/bw/bw/while/Switch_4:1*
T0*
_output_shapes
:	@

#bidirectional_rnn/bw/bw/while/add/yConst'^bidirectional_rnn/bw/bw/while/Identity*
_output_shapes
: *
dtype0*
value	B :

!bidirectional_rnn/bw/bw/while/addAdd&bidirectional_rnn/bw/bw/while/Identity#bidirectional_rnn/bw/bw/while/add/y*
T0*
_output_shapes
: 

/bidirectional_rnn/bw/bw/while/TensorArrayReadV3TensorArrayReadV35bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter(bidirectional_rnn/bw/bw/while/Identity_17bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1* 
_output_shapes
:
*
dtype0
ő
5bidirectional_rnn/bw/bw/while/TensorArrayReadV3/EnterEnter%bidirectional_rnn/bw/bw/TensorArray_1*
parallel_iterations *
T0*
_output_shapes
:*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context*
is_constant(
 
7bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1EnterRbidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context*
parallel_iterations *
_output_shapes
: *
T0*
is_constant(
š
*bidirectional_rnn/bw/bw/while/GreaterEqualGreaterEqual(bidirectional_rnn/bw/bw/while/Identity_10bidirectional_rnn/bw/bw/while/GreaterEqual/Enter*
_output_shapes
:*
T0
ě
0bidirectional_rnn/bw/bw/while/GreaterEqual/EnterEnter#bidirectional_rnn/bw/bw/CheckSeqLen*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context*
parallel_iterations *
_output_shapes
:*
is_constant(*
T0
Ý
Lbidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
valueB"@     *
dtype0*>
_class4
20loc:@bidirectional_rnn/bw/basic_lstm_cell/kernel
Ď
Jbidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *ňę­˝*>
_class4
20loc:@bidirectional_rnn/bw/basic_lstm_cell/kernel
Ď
Jbidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/maxConst*
dtype0*>
_class4
20loc:@bidirectional_rnn/bw/basic_lstm_cell/kernel*
_output_shapes
: *
valueB
 *ňę­=
Ä
Tbidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformLbidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/shape*>
_class4
20loc:@bidirectional_rnn/bw/basic_lstm_cell/kernel*

seed * 
_output_shapes
:
Ŕ*
dtype0*
seed2 *
T0
Ę
Jbidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/subSubJbidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/maxJbidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *>
_class4
20loc:@bidirectional_rnn/bw/basic_lstm_cell/kernel
Ţ
Jbidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/mulMulTbidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformJbidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
Ŕ*>
_class4
20loc:@bidirectional_rnn/bw/basic_lstm_cell/kernel*
T0
Đ
Fbidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniformAddJbidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/mulJbidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/min*
T0* 
_output_shapes
:
Ŕ*>
_class4
20loc:@bidirectional_rnn/bw/basic_lstm_cell/kernel
ă
+bidirectional_rnn/bw/basic_lstm_cell/kernel
VariableV2*>
_class4
20loc:@bidirectional_rnn/bw/basic_lstm_cell/kernel*
dtype0* 
_output_shapes
:
Ŕ*
shape:
Ŕ*
shared_name *
	container 
Ĺ
2bidirectional_rnn/bw/basic_lstm_cell/kernel/AssignAssign+bidirectional_rnn/bw/basic_lstm_cell/kernelFbidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform*>
_class4
20loc:@bidirectional_rnn/bw/basic_lstm_cell/kernel*
use_locking(* 
_output_shapes
:
Ŕ*
T0*
validate_shape(

0bidirectional_rnn/bw/basic_lstm_cell/kernel/readIdentity+bidirectional_rnn/bw/basic_lstm_cell/kernel* 
_output_shapes
:
Ŕ*
T0
Č
;bidirectional_rnn/bw/basic_lstm_cell/bias/Initializer/zerosConst*
valueB*    *<
_class2
0.loc:@bidirectional_rnn/bw/basic_lstm_cell/bias*
_output_shapes	
:*
dtype0
Ő
)bidirectional_rnn/bw/basic_lstm_cell/bias
VariableV2*
_output_shapes	
:*
shape:*
dtype0*
	container *<
_class2
0.loc:@bidirectional_rnn/bw/basic_lstm_cell/bias*
shared_name 
Ż
0bidirectional_rnn/bw/basic_lstm_cell/bias/AssignAssign)bidirectional_rnn/bw/basic_lstm_cell/bias;bidirectional_rnn/bw/basic_lstm_cell/bias/Initializer/zeros*
validate_shape(*
T0*<
_class2
0.loc:@bidirectional_rnn/bw/basic_lstm_cell/bias*
_output_shapes	
:*
use_locking(

.bidirectional_rnn/bw/basic_lstm_cell/bias/readIdentity)bidirectional_rnn/bw/basic_lstm_cell/bias*
T0*
_output_shapes	
:

3bidirectional_rnn/bw/bw/while/basic_lstm_cell/ConstConst'^bidirectional_rnn/bw/bw/while/Identity*
_output_shapes
: *
dtype0*
value	B :
¤
9bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat/axisConst'^bidirectional_rnn/bw/bw/while/Identity*
_output_shapes
: *
value	B :*
dtype0

4bidirectional_rnn/bw/bw/while/basic_lstm_cell/concatConcatV2/bidirectional_rnn/bw/bw/while/TensorArrayReadV3(bidirectional_rnn/bw/bw/while/Identity_49bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat/axis*
T0* 
_output_shapes
:
Ŕ*

Tidx0*
N

4bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMulMatMul4bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat:bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul/Enter*
transpose_a( * 
_output_shapes
:
*
T0*
transpose_b( 

:bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul/EnterEnter0bidirectional_rnn/bw/basic_lstm_cell/kernel/read*
is_constant(*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context*
parallel_iterations *
T0* 
_output_shapes
:
Ŕ
ő
5bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAddBiasAdd4bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul;bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd/Enter*
T0* 
_output_shapes
:
*
data_formatNHWC

;bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd/EnterEnter.bidirectional_rnn/bw/basic_lstm_cell/bias/read*
parallel_iterations *
T0*
_output_shapes	
:*
is_constant(*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context
 
5bidirectional_rnn/bw/bw/while/basic_lstm_cell/Const_1Const'^bidirectional_rnn/bw/bw/while/Identity*
_output_shapes
: *
value	B :*
dtype0

3bidirectional_rnn/bw/bw/while/basic_lstm_cell/splitSplit3bidirectional_rnn/bw/bw/while/basic_lstm_cell/Const5bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd*
T0*@
_output_shapes.
,:	@:	@:	@:	@*
	num_split
Ł
5bidirectional_rnn/bw/bw/while/basic_lstm_cell/Const_2Const'^bidirectional_rnn/bw/bw/while/Identity*
_output_shapes
: *
dtype0*
valueB
 *    
Đ
1bidirectional_rnn/bw/bw/while/basic_lstm_cell/AddAdd5bidirectional_rnn/bw/bw/while/basic_lstm_cell/split:25bidirectional_rnn/bw/bw/while/basic_lstm_cell/Const_2*
_output_shapes
:	@*
T0

5bidirectional_rnn/bw/bw/while/basic_lstm_cell/SigmoidSigmoid1bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add*
_output_shapes
:	@*
T0
Ă
1bidirectional_rnn/bw/bw/while/basic_lstm_cell/MulMul(bidirectional_rnn/bw/bw/while/Identity_35bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid*
_output_shapes
:	@*
T0
Ą
7bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid_1Sigmoid3bidirectional_rnn/bw/bw/while/basic_lstm_cell/split*
T0*
_output_shapes
:	@

2bidirectional_rnn/bw/bw/while/basic_lstm_cell/TanhTanh5bidirectional_rnn/bw/bw/while/basic_lstm_cell/split:1*
T0*
_output_shapes
:	@
Ń
3bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1Mul7bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid_12bidirectional_rnn/bw/bw/while/basic_lstm_cell/Tanh*
_output_shapes
:	@*
T0
Ě
3bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_1Add1bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul3bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1*
_output_shapes
:	@*
T0

4bidirectional_rnn/bw/bw/while/basic_lstm_cell/Tanh_1Tanh3bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_1*
T0*
_output_shapes
:	@
Ł
7bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid_2Sigmoid5bidirectional_rnn/bw/bw/while/basic_lstm_cell/split:3*
T0*
_output_shapes
:	@
Ó
3bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2Mul4bidirectional_rnn/bw/bw/while/basic_lstm_cell/Tanh_17bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid_2*
T0*
_output_shapes
:	@

*bidirectional_rnn/bw/bw/while/dropout/rateConst'^bidirectional_rnn/bw/bw/while/Identity*
dtype0*
valueB
 *   ?*
_output_shapes
: 
Ľ
+bidirectional_rnn/bw/bw/while/dropout/ShapeConst'^bidirectional_rnn/bw/bw/while/Identity*
_output_shapes
:*
dtype0*
valueB"   @   
Ś
8bidirectional_rnn/bw/bw/while/dropout/random_uniform/minConst'^bidirectional_rnn/bw/bw/while/Identity*
valueB
 *    *
_output_shapes
: *
dtype0
Ś
8bidirectional_rnn/bw/bw/while/dropout/random_uniform/maxConst'^bidirectional_rnn/bw/bw/while/Identity*
dtype0*
_output_shapes
: *
valueB
 *  ?
Đ
Bbidirectional_rnn/bw/bw/while/dropout/random_uniform/RandomUniformRandomUniform+bidirectional_rnn/bw/bw/while/dropout/Shape*
dtype0*
T0*

seed *
_output_shapes
:	@*
seed2 
Ô
8bidirectional_rnn/bw/bw/while/dropout/random_uniform/subSub8bidirectional_rnn/bw/bw/while/dropout/random_uniform/max8bidirectional_rnn/bw/bw/while/dropout/random_uniform/min*
_output_shapes
: *
T0
ç
8bidirectional_rnn/bw/bw/while/dropout/random_uniform/mulMulBbidirectional_rnn/bw/bw/while/dropout/random_uniform/RandomUniform8bidirectional_rnn/bw/bw/while/dropout/random_uniform/sub*
_output_shapes
:	@*
T0
Ů
4bidirectional_rnn/bw/bw/while/dropout/random_uniformAdd8bidirectional_rnn/bw/bw/while/dropout/random_uniform/mul8bidirectional_rnn/bw/bw/while/dropout/random_uniform/min*
_output_shapes
:	@*
T0

+bidirectional_rnn/bw/bw/while/dropout/sub/xConst'^bidirectional_rnn/bw/bw/while/Identity*
dtype0*
valueB
 *  ?*
_output_shapes
: 
Ş
)bidirectional_rnn/bw/bw/while/dropout/subSub+bidirectional_rnn/bw/bw/while/dropout/sub/x*bidirectional_rnn/bw/bw/while/dropout/rate*
_output_shapes
: *
T0

/bidirectional_rnn/bw/bw/while/dropout/truediv/xConst'^bidirectional_rnn/bw/bw/while/Identity*
dtype0*
_output_shapes
: *
valueB
 *  ?
ľ
-bidirectional_rnn/bw/bw/while/dropout/truedivRealDiv/bidirectional_rnn/bw/bw/while/dropout/truediv/x)bidirectional_rnn/bw/bw/while/dropout/sub*
_output_shapes
: *
T0
Î
2bidirectional_rnn/bw/bw/while/dropout/GreaterEqualGreaterEqual4bidirectional_rnn/bw/bw/while/dropout/random_uniform*bidirectional_rnn/bw/bw/while/dropout/rate*
T0*
_output_shapes
:	@
ž
)bidirectional_rnn/bw/bw/while/dropout/mulMul3bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2-bidirectional_rnn/bw/bw/while/dropout/truediv*
_output_shapes
:	@*
T0
Ż
*bidirectional_rnn/bw/bw/while/dropout/CastCast2bidirectional_rnn/bw/bw/while/dropout/GreaterEqual*

DstT0*
_output_shapes
:	@*
Truncate( *

SrcT0

ł
+bidirectional_rnn/bw/bw/while/dropout/mul_1Mul)bidirectional_rnn/bw/bw/while/dropout/mul*bidirectional_rnn/bw/bw/while/dropout/Cast*
_output_shapes
:	@*
T0

$bidirectional_rnn/bw/bw/while/SelectSelect*bidirectional_rnn/bw/bw/while/GreaterEqual*bidirectional_rnn/bw/bw/while/Select/Enter+bidirectional_rnn/bw/bw/while/dropout/mul_1*
T0*>
_class4
20loc:@bidirectional_rnn/bw/bw/while/dropout/mul_1*
_output_shapes
:	@
§
*bidirectional_rnn/bw/bw/while/Select/EnterEnterbidirectional_rnn/bw/bw/zeros*
T0*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
parallel_iterations *>
_class4
20loc:@bidirectional_rnn/bw/bw/while/dropout/mul_1*
_output_shapes
:	@
­
&bidirectional_rnn/bw/bw/while/Select_1Select*bidirectional_rnn/bw/bw/while/GreaterEqual(bidirectional_rnn/bw/bw/while/Identity_33bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_1*F
_class<
:8loc:@bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_1*
T0*
_output_shapes
:	@
­
&bidirectional_rnn/bw/bw/while/Select_2Select*bidirectional_rnn/bw/bw/while/GreaterEqual(bidirectional_rnn/bw/bw/while/Identity_43bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2*
T0*
_output_shapes
:	@*F
_class<
:8loc:@bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2
ű
Abidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Gbidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3/Enter(bidirectional_rnn/bw/bw/while/Identity_1$bidirectional_rnn/bw/bw/while/Select(bidirectional_rnn/bw/bw/while/Identity_2*
_output_shapes
: *
T0*>
_class4
20loc:@bidirectional_rnn/bw/bw/while/dropout/mul_1
Ĺ
Gbidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnter#bidirectional_rnn/bw/bw/TensorArray*>
_class4
20loc:@bidirectional_rnn/bw/bw/while/dropout/mul_1*
_output_shapes
:*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context*
T0*
parallel_iterations *
is_constant(

%bidirectional_rnn/bw/bw/while/add_1/yConst'^bidirectional_rnn/bw/bw/while/Identity*
dtype0*
_output_shapes
: *
value	B :

#bidirectional_rnn/bw/bw/while/add_1Add(bidirectional_rnn/bw/bw/while/Identity_1%bidirectional_rnn/bw/bw/while/add_1/y*
T0*
_output_shapes
: 

+bidirectional_rnn/bw/bw/while/NextIterationNextIteration!bidirectional_rnn/bw/bw/while/add*
T0*
_output_shapes
: 

-bidirectional_rnn/bw/bw/while/NextIteration_1NextIteration#bidirectional_rnn/bw/bw/while/add_1*
T0*
_output_shapes
: 
˘
-bidirectional_rnn/bw/bw/while/NextIteration_2NextIterationAbidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3*
_output_shapes
: *
T0

-bidirectional_rnn/bw/bw/while/NextIteration_3NextIteration&bidirectional_rnn/bw/bw/while/Select_1*
_output_shapes
:	@*
T0

-bidirectional_rnn/bw/bw/while/NextIteration_4NextIteration&bidirectional_rnn/bw/bw/while/Select_2*
_output_shapes
:	@*
T0
q
"bidirectional_rnn/bw/bw/while/ExitExit$bidirectional_rnn/bw/bw/while/Switch*
_output_shapes
: *
T0
u
$bidirectional_rnn/bw/bw/while/Exit_1Exit&bidirectional_rnn/bw/bw/while/Switch_1*
T0*
_output_shapes
: 
u
$bidirectional_rnn/bw/bw/while/Exit_2Exit&bidirectional_rnn/bw/bw/while/Switch_2*
_output_shapes
: *
T0
~
$bidirectional_rnn/bw/bw/while/Exit_3Exit&bidirectional_rnn/bw/bw/while/Switch_3*
T0*
_output_shapes
:	@
~
$bidirectional_rnn/bw/bw/while/Exit_4Exit&bidirectional_rnn/bw/bw/while/Switch_4*
T0*
_output_shapes
:	@
ę
:bidirectional_rnn/bw/bw/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3#bidirectional_rnn/bw/bw/TensorArray$bidirectional_rnn/bw/bw/while/Exit_2*6
_class,
*(loc:@bidirectional_rnn/bw/bw/TensorArray*
_output_shapes
: 
Ž
4bidirectional_rnn/bw/bw/TensorArrayStack/range/startConst*
dtype0*
_output_shapes
: *6
_class,
*(loc:@bidirectional_rnn/bw/bw/TensorArray*
value	B : 
Ž
4bidirectional_rnn/bw/bw/TensorArrayStack/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: *6
_class,
*(loc:@bidirectional_rnn/bw/bw/TensorArray
Č
.bidirectional_rnn/bw/bw/TensorArrayStack/rangeRange4bidirectional_rnn/bw/bw/TensorArrayStack/range/start:bidirectional_rnn/bw/bw/TensorArrayStack/TensorArraySizeV34bidirectional_rnn/bw/bw/TensorArrayStack/range/delta*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tidx0*6
_class,
*(loc:@bidirectional_rnn/bw/bw/TensorArray
ß
<bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3#bidirectional_rnn/bw/bw/TensorArray.bidirectional_rnn/bw/bw/TensorArrayStack/range$bidirectional_rnn/bw/bw/while/Exit_2*
element_shape:	@*
dtype0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@*6
_class,
*(loc:@bidirectional_rnn/bw/bw/TensorArray
i
bidirectional_rnn/bw/bw/Const_3Const*
valueB:@*
dtype0*
_output_shapes
:
`
bidirectional_rnn/bw/bw/Rank_3Const*
dtype0*
value	B :*
_output_shapes
: 
g
%bidirectional_rnn/bw/bw/range_3/startConst*
dtype0*
_output_shapes
: *
value	B :
g
%bidirectional_rnn/bw/bw/range_3/deltaConst*
dtype0*
_output_shapes
: *
value	B :
ž
bidirectional_rnn/bw/bw/range_3Range%bidirectional_rnn/bw/bw/range_3/startbidirectional_rnn/bw/bw/Rank_3%bidirectional_rnn/bw/bw/range_3/delta*

Tidx0*
_output_shapes
:
z
)bidirectional_rnn/bw/bw/concat_2/values_0Const*
dtype0*
valueB"       *
_output_shapes
:
g
%bidirectional_rnn/bw/bw/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Ů
 bidirectional_rnn/bw/bw/concat_2ConcatV2)bidirectional_rnn/bw/bw/concat_2/values_0bidirectional_rnn/bw/bw/range_3%bidirectional_rnn/bw/bw/concat_2/axis*

Tidx0*
T0*
N*
_output_shapes
:
Ô
#bidirectional_rnn/bw/bw/transpose_1	Transpose<bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3 bidirectional_rnn/bw/bw/concat_2*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
Tperm0
š
ReverseSequenceReverseSequence#bidirectional_rnn/bw/bw/transpose_1Placeholder_2*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
seq_dim*

Tlen0*
	batch_dim *
T0
M
concat/axisConst*
dtype0*
value	B :*
_output_shapes
: 
˘
concatConcatV2#bidirectional_rnn/fw/fw/transpose_1ReverseSequenceconcat/axis*

Tidx0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙*
N*
T0
b
Reshape/shapeConst*!
valueB"         *
_output_shapes
:*
dtype0
f
ReshapeReshapeconcatReshape/shape*
T0*$
_output_shapes
:*
Tshape0

(Weights/Initializer/random_uniform/shapeConst*
_class
loc:@Weights*
_output_shapes
:*
valueB"      *
dtype0

&Weights/Initializer/random_uniform/minConst*
dtype0*
valueB
 *ý[ž*
_output_shapes
: *
_class
loc:@Weights

&Weights/Initializer/random_uniform/maxConst*
_class
loc:@Weights*
valueB
 *ý[>*
dtype0*
_output_shapes
: 
×
0Weights/Initializer/random_uniform/RandomUniformRandomUniform(Weights/Initializer/random_uniform/shape*
T0*
dtype0*
seed2 *
_output_shapes
:	*
_class
loc:@Weights*

seed 
ş
&Weights/Initializer/random_uniform/subSub&Weights/Initializer/random_uniform/max&Weights/Initializer/random_uniform/min*
T0*
_output_shapes
: *
_class
loc:@Weights
Í
&Weights/Initializer/random_uniform/mulMul0Weights/Initializer/random_uniform/RandomUniform&Weights/Initializer/random_uniform/sub*
_class
loc:@Weights*
T0*
_output_shapes
:	
ż
"Weights/Initializer/random_uniformAdd&Weights/Initializer/random_uniform/mul&Weights/Initializer/random_uniform/min*
T0*
_output_shapes
:	*
_class
loc:@Weights

Weights
VariableV2*
	container *
shared_name *
shape:	*
_class
loc:@Weights*
dtype0*
_output_shapes
:	
´
Weights/AssignAssignWeights"Weights/Initializer/random_uniform*
_output_shapes
:	*
T0*
_class
loc:@Weights*
use_locking(*
validate_shape(
g
Weights/readIdentityWeights*
T0*
_class
loc:@Weights*
_output_shapes
:	

%Bias/Initializer/random_uniform/shapeConst*
_class
	loc:@Bias*
dtype0*
_output_shapes
:*
valueB:

#Bias/Initializer/random_uniform/minConst*
valueB
 *qÄż*
_output_shapes
: *
_class
	loc:@Bias*
dtype0

#Bias/Initializer/random_uniform/maxConst*
_class
	loc:@Bias*
valueB
 *qÄ?*
_output_shapes
: *
dtype0
É
-Bias/Initializer/random_uniform/RandomUniformRandomUniform%Bias/Initializer/random_uniform/shape*

seed *
_output_shapes
:*
T0*
dtype0*
_class
	loc:@Bias*
seed2 
Ž
#Bias/Initializer/random_uniform/subSub#Bias/Initializer/random_uniform/max#Bias/Initializer/random_uniform/min*
_class
	loc:@Bias*
_output_shapes
: *
T0
ź
#Bias/Initializer/random_uniform/mulMul-Bias/Initializer/random_uniform/RandomUniform#Bias/Initializer/random_uniform/sub*
_class
	loc:@Bias*
_output_shapes
:*
T0
Ž
Bias/Initializer/random_uniformAdd#Bias/Initializer/random_uniform/mul#Bias/Initializer/random_uniform/min*
_class
	loc:@Bias*
T0*
_output_shapes
:

Bias
VariableV2*
shared_name *
_class
	loc:@Bias*
dtype0*
	container *
_output_shapes
:*
shape:
Ł
Bias/AssignAssignBiasBias/Initializer/random_uniform*
T0*
_class
	loc:@Bias*
validate_shape(*
_output_shapes
:*
use_locking(
Y
	Bias/readIdentityBias*
T0*
_class
	loc:@Bias*
_output_shapes
:
v
MatMulBatchMatMulV2ReshapeWeights/read*
T0*#
_output_shapes
:*
adj_x( *
adj_y( 
K
addAddMatMul	Bias/read*
T0*#
_output_shapes
:
R
ArgMax/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
p
ArgMaxArgMaxaddArgMax/dimension*
output_type0	*
_output_shapes
:	*
T0*

Tidx0
R
one_hot/on_valueConst*
_output_shapes
: *
value	B :*
dtype0
S
one_hot/off_valueConst*
dtype0*
value	B : *
_output_shapes
: 
O
one_hot/depthConst*
value	B :*
_output_shapes
: *
dtype0

one_hotOneHotPlaceholder_1one_hot/depthone_hot/on_valueone_hot/off_value*
axis˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
TI0*
T0
d
Reshape_1/shapeConst*!
valueB"         *
_output_shapes
:*
dtype0
j
	Reshape_1Reshapeone_hotReshape_1/shape*
T0*#
_output_shapes
:*
Tshape0

9softmax_cross_entropy_with_logits_sg/labels_stop_gradientStopGradient	Reshape_1*
T0*#
_output_shapes
:
š
)softmax_cross_entropy_with_logits_sg/CastCast9softmax_cross_entropy_with_logits_sg/labels_stop_gradient*

DstT0*

SrcT0*#
_output_shapes
:*
Truncate( 
k
)softmax_cross_entropy_with_logits_sg/RankConst*
_output_shapes
: *
value	B :*
dtype0

*softmax_cross_entropy_with_logits_sg/ShapeConst*!
valueB"         *
_output_shapes
:*
dtype0
m
+softmax_cross_entropy_with_logits_sg/Rank_1Const*
_output_shapes
: *
value	B :*
dtype0

,softmax_cross_entropy_with_logits_sg/Shape_1Const*
dtype0*!
valueB"         *
_output_shapes
:
l
*softmax_cross_entropy_with_logits_sg/Sub/yConst*
_output_shapes
: *
value	B :*
dtype0
Š
(softmax_cross_entropy_with_logits_sg/SubSub+softmax_cross_entropy_with_logits_sg/Rank_1*softmax_cross_entropy_with_logits_sg/Sub/y*
_output_shapes
: *
T0

0softmax_cross_entropy_with_logits_sg/Slice/beginPack(softmax_cross_entropy_with_logits_sg/Sub*

axis *
T0*
_output_shapes
:*
N
y
/softmax_cross_entropy_with_logits_sg/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:
ö
*softmax_cross_entropy_with_logits_sg/SliceSlice,softmax_cross_entropy_with_logits_sg/Shape_10softmax_cross_entropy_with_logits_sg/Slice/begin/softmax_cross_entropy_with_logits_sg/Slice/size*
_output_shapes
:*
T0*
Index0

4softmax_cross_entropy_with_logits_sg/concat/values_0Const*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
r
0softmax_cross_entropy_with_logits_sg/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 

+softmax_cross_entropy_with_logits_sg/concatConcatV24softmax_cross_entropy_with_logits_sg/concat/values_0*softmax_cross_entropy_with_logits_sg/Slice0softmax_cross_entropy_with_logits_sg/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
˘
,softmax_cross_entropy_with_logits_sg/ReshapeReshapeadd+softmax_cross_entropy_with_logits_sg/concat* 
_output_shapes
:
 *
T0*
Tshape0
m
+softmax_cross_entropy_with_logits_sg/Rank_2Const*
_output_shapes
: *
value	B :*
dtype0

,softmax_cross_entropy_with_logits_sg/Shape_2Const*
dtype0*
_output_shapes
:*!
valueB"         
n
,softmax_cross_entropy_with_logits_sg/Sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
­
*softmax_cross_entropy_with_logits_sg/Sub_1Sub+softmax_cross_entropy_with_logits_sg/Rank_2,softmax_cross_entropy_with_logits_sg/Sub_1/y*
T0*
_output_shapes
: 
 
2softmax_cross_entropy_with_logits_sg/Slice_1/beginPack*softmax_cross_entropy_with_logits_sg/Sub_1*
N*
T0*

axis *
_output_shapes
:
{
1softmax_cross_entropy_with_logits_sg/Slice_1/sizeConst*
dtype0*
valueB:*
_output_shapes
:
ü
,softmax_cross_entropy_with_logits_sg/Slice_1Slice,softmax_cross_entropy_with_logits_sg/Shape_22softmax_cross_entropy_with_logits_sg/Slice_1/begin1softmax_cross_entropy_with_logits_sg/Slice_1/size*
T0*
_output_shapes
:*
Index0

6softmax_cross_entropy_with_logits_sg/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙
t
2softmax_cross_entropy_with_logits_sg/concat_1/axisConst*
dtype0*
value	B : *
_output_shapes
: 

-softmax_cross_entropy_with_logits_sg/concat_1ConcatV26softmax_cross_entropy_with_logits_sg/concat_1/values_0,softmax_cross_entropy_with_logits_sg/Slice_12softmax_cross_entropy_with_logits_sg/concat_1/axis*
_output_shapes
:*

Tidx0*
N*
T0
Ě
.softmax_cross_entropy_with_logits_sg/Reshape_1Reshape)softmax_cross_entropy_with_logits_sg/Cast-softmax_cross_entropy_with_logits_sg/concat_1* 
_output_shapes
:
 *
Tshape0*
T0
Ö
$softmax_cross_entropy_with_logits_sgSoftmaxCrossEntropyWithLogits,softmax_cross_entropy_with_logits_sg/Reshape.softmax_cross_entropy_with_logits_sg/Reshape_1*(
_output_shapes
: :
 *
T0
n
,softmax_cross_entropy_with_logits_sg/Sub_2/yConst*
_output_shapes
: *
dtype0*
value	B :
Ť
*softmax_cross_entropy_with_logits_sg/Sub_2Sub)softmax_cross_entropy_with_logits_sg/Rank,softmax_cross_entropy_with_logits_sg/Sub_2/y*
T0*
_output_shapes
: 
|
2softmax_cross_entropy_with_logits_sg/Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:

1softmax_cross_entropy_with_logits_sg/Slice_2/sizePack*softmax_cross_entropy_with_logits_sg/Sub_2*

axis *
T0*
N*
_output_shapes
:
ú
,softmax_cross_entropy_with_logits_sg/Slice_2Slice*softmax_cross_entropy_with_logits_sg/Shape2softmax_cross_entropy_with_logits_sg/Slice_2/begin1softmax_cross_entropy_with_logits_sg/Slice_2/size*
T0*
Index0*
_output_shapes
:
Ĺ
.softmax_cross_entropy_with_logits_sg/Reshape_2Reshape$softmax_cross_entropy_with_logits_sg,softmax_cross_entropy_with_logits_sg/Slice_2*
Tshape0*
_output_shapes
:	*
T0
`
gradients/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      
X
gradients/grad_ys_0Const*
_output_shapes
: *
dtype0*
valueB
 *  ?
x
gradients/FillFillgradients/Shapegradients/grad_ys_0*
_output_shapes
:	*
T0*

index_type0
S
gradients/f_countConst*
value	B : *
dtype0*
_output_shapes
: 
ť
gradients/f_count_1Entergradients/f_count*
parallel_iterations *
T0*
_output_shapes
: *;

frame_name-+bidirectional_rnn/fw/fw/while/while_context*
is_constant( 
r
gradients/MergeMergegradients/f_count_1gradients/NextIteration*
T0*
N*
_output_shapes
: : 
v
gradients/SwitchSwitchgradients/Merge&bidirectional_rnn/fw/fw/while/LoopCond*
T0*
_output_shapes
: : 
z
gradients/Add/yConst'^bidirectional_rnn/fw/fw/while/Identity*
_output_shapes
: *
dtype0*
value	B :
Z
gradients/AddAddgradients/Switch:1gradients/Add/y*
_output_shapes
: *
T0
Ł	
gradients/NextIterationNextIterationgradients/AddI^gradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPushV2o^gradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2Y^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2S^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul/StackPushV2U^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPushV2S^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul/StackPushV2U^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPushV2Q^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul/StackPushV2S^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul_1/StackPushV2K^gradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul/StackPushV2M^gradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul_1/StackPushV2I^gradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/StackPushV2K^gradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/StackPushV2*
T0*
_output_shapes
: 
N
gradients/f_count_2Exitgradients/Switch*
_output_shapes
: *
T0
S
gradients/b_countConst*
value	B :*
_output_shapes
: *
dtype0
Ç
gradients/b_count_1Entergradients/f_count_2*
parallel_iterations *
is_constant( *E

frame_name75gradients/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
: *
T0
v
gradients/Merge_1Mergegradients/b_count_1gradients/NextIteration_1*
T0*
_output_shapes
: : *
N
x
gradients/GreaterEqualGreaterEqualgradients/Merge_1gradients/GreaterEqual/Enter*
_output_shapes
: *
T0
Î
gradients/GreaterEqual/EnterEntergradients/b_count*E

frame_name75gradients/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
: *
T0
O
gradients/b_count_2LoopCondgradients/GreaterEqual*
_output_shapes
: 
g
gradients/Switch_1Switchgradients/Merge_1gradients/b_count_2*
_output_shapes
: : *
T0
i
gradients/SubSubgradients/Switch_1:1gradients/GreaterEqual/Enter*
_output_shapes
: *
T0
Ć
gradients/NextIteration_1NextIterationgradients/Subj^gradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_sync*
_output_shapes
: *
T0
P
gradients/b_count_3Exitgradients/Switch_1*
T0*
_output_shapes
: 
U
gradients/f_count_3Const*
dtype0*
_output_shapes
: *
value	B : 
˝
gradients/f_count_4Entergradients/f_count_3*
_output_shapes
: *;

frame_name-+bidirectional_rnn/bw/bw/while/while_context*
T0*
is_constant( *
parallel_iterations 
v
gradients/Merge_2Mergegradients/f_count_4gradients/NextIteration_2*
N*
_output_shapes
: : *
T0
z
gradients/Switch_2Switchgradients/Merge_2&bidirectional_rnn/bw/bw/while/LoopCond*
T0*
_output_shapes
: : 
|
gradients/Add_1/yConst'^bidirectional_rnn/bw/bw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
`
gradients/Add_1Addgradients/Switch_2:1gradients/Add_1/y*
_output_shapes
: *
T0
§	
gradients/NextIteration_2NextIterationgradients/Add_1I^gradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPushV2o^gradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2Y^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2S^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul/StackPushV2U^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPushV2S^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul/StackPushV2U^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPushV2Q^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul/StackPushV2S^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul_1/StackPushV2K^gradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul/StackPushV2M^gradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul_1/StackPushV2I^gradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/StackPushV2K^gradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/StackPushV2*
T0*
_output_shapes
: 
P
gradients/f_count_5Exitgradients/Switch_2*
T0*
_output_shapes
: 
U
gradients/b_count_4Const*
value	B :*
_output_shapes
: *
dtype0
Ç
gradients/b_count_5Entergradients/f_count_5*
_output_shapes
: *
T0*
parallel_iterations *E

frame_name75gradients/bidirectional_rnn/bw/bw/while/while_context*
is_constant( 
v
gradients/Merge_3Mergegradients/b_count_5gradients/NextIteration_3*
N*
_output_shapes
: : *
T0
|
gradients/GreaterEqual_1GreaterEqualgradients/Merge_3gradients/GreaterEqual_1/Enter*
T0*
_output_shapes
: 
Ň
gradients/GreaterEqual_1/EnterEntergradients/b_count_4*
is_constant(*
T0*
parallel_iterations *
_output_shapes
: *E

frame_name75gradients/bidirectional_rnn/bw/bw/while/while_context
Q
gradients/b_count_6LoopCondgradients/GreaterEqual_1*
_output_shapes
: 
g
gradients/Switch_3Switchgradients/Merge_3gradients/b_count_6*
_output_shapes
: : *
T0
m
gradients/Sub_1Subgradients/Switch_3:1gradients/GreaterEqual_1/Enter*
_output_shapes
: *
T0
Č
gradients/NextIteration_3NextIterationgradients/Sub_1j^gradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_sync*
T0*
_output_shapes
: 
P
gradients/b_count_7Exitgradients/Switch_3*
_output_shapes
: *
T0

Cgradients/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB: 
Ú
Egradients/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeReshapegradients/FillCgradients/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/Shape*
_output_shapes

: *
Tshape0*
T0
t
gradients/zeros_like	ZerosLike&softmax_cross_entropy_with_logits_sg:1* 
_output_shapes
:
 *
T0

Bgradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dimConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 

>gradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims
ExpandDimsEgradients/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeBgradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dim*
T0* 
_output_shapes
:
 *

Tdim0
Ń
7gradients/softmax_cross_entropy_with_logits_sg_grad/mulMul>gradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims&softmax_cross_entropy_with_logits_sg:1*
T0* 
_output_shapes
:
 
Ľ
>gradients/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax
LogSoftmax,softmax_cross_entropy_with_logits_sg/Reshape*
T0* 
_output_shapes
:
 
Š
7gradients/softmax_cross_entropy_with_logits_sg_grad/NegNeg>gradients/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax* 
_output_shapes
:
 *
T0

Dgradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙

@gradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1
ExpandDimsEgradients/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeDgradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dim*

Tdim0* 
_output_shapes
:
 *
T0
ć
9gradients/softmax_cross_entropy_with_logits_sg_grad/mul_1Mul@gradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_17gradients/softmax_cross_entropy_with_logits_sg_grad/Neg*
T0* 
_output_shapes
:
 
Â
Dgradients/softmax_cross_entropy_with_logits_sg_grad/tuple/group_depsNoOp8^gradients/softmax_cross_entropy_with_logits_sg_grad/mul:^gradients/softmax_cross_entropy_with_logits_sg_grad/mul_1
Ď
Lgradients/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependencyIdentity7gradients/softmax_cross_entropy_with_logits_sg_grad/mulE^gradients/softmax_cross_entropy_with_logits_sg_grad/tuple/group_deps*J
_class@
><loc:@gradients/softmax_cross_entropy_with_logits_sg_grad/mul*
T0* 
_output_shapes
:
 
Ő
Ngradients/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependency_1Identity9gradients/softmax_cross_entropy_with_logits_sg_grad/mul_1E^gradients/softmax_cross_entropy_with_logits_sg_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/softmax_cross_entropy_with_logits_sg_grad/mul_1* 
_output_shapes
:
 

Agradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/ShapeConst*
dtype0*
_output_shapes
:*!
valueB"         

Cgradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/ReshapeReshapeLgradients/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependencyAgradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/Shape*#
_output_shapes
:*
T0*
Tshape0
m
gradients/add_grad/ShapeConst*
dtype0*
_output_shapes
:*!
valueB"         
d
gradients/add_grad/Shape_1Const*
valueB:*
_output_shapes
:*
dtype0
´
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
×
gradients/add_grad/SumSumCgradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape(gradients/add_grad/BroadcastGradientArgs*

Tidx0*#
_output_shapes
:*
T0*
	keep_dims( 

gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
Tshape0*#
_output_shapes
:*
T0
Ň
gradients/add_grad/Sum_1SumCgradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape*gradients/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:

gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
Ö
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*-
_class#
!loc:@gradients/add_grad/Reshape*
T0*#
_output_shapes
:
Ó
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
_output_shapes
:
ą
gradients/MatMul_grad/MatMulBatchMatMulV2+gradients/add_grad/tuple/control_dependencyWeights/read*$
_output_shapes
:*
adj_y(*
adj_x( *
T0
Ž
gradients/MatMul_grad/MatMul_1BatchMatMulV2Reshape+gradients/add_grad/tuple/control_dependency*
T0*
adj_x(*$
_output_shapes
:*
adj_y( 
p
gradients/MatMul_grad/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"         
n
gradients/MatMul_grad/Shape_1Const*
dtype0*
valueB"      *
_output_shapes
:
s
)gradients/MatMul_grad/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
~
+gradients/MatMul_grad/strided_slice/stack_1Const*
_output_shapes
:*
valueB:
ţ˙˙˙˙˙˙˙˙*
dtype0
u
+gradients/MatMul_grad/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
ë
#gradients/MatMul_grad/strided_sliceStridedSlicegradients/MatMul_grad/Shape)gradients/MatMul_grad/strided_slice/stack+gradients/MatMul_grad/strided_slice/stack_1+gradients/MatMul_grad/strided_slice/stack_2*
T0*
shrink_axis_mask *

begin_mask*
end_mask *
new_axis_mask *
_output_shapes
:*
ellipsis_mask *
Index0
u
+gradients/MatMul_grad/strided_slice_1/stackConst*
dtype0*
valueB: *
_output_shapes
:

-gradients/MatMul_grad/strided_slice_1/stack_1Const*
valueB:
ţ˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
w
-gradients/MatMul_grad/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
ó
%gradients/MatMul_grad/strided_slice_1StridedSlicegradients/MatMul_grad/Shape_1+gradients/MatMul_grad/strided_slice_1/stack-gradients/MatMul_grad/strided_slice_1/stack_1-gradients/MatMul_grad/strided_slice_1/stack_2*

begin_mask*
new_axis_mask *
_output_shapes
: *
Index0*
shrink_axis_mask *
end_mask *
ellipsis_mask *
T0
Í
+gradients/MatMul_grad/BroadcastGradientArgsBroadcastGradientArgs#gradients/MatMul_grad/strided_slice%gradients/MatMul_grad/strided_slice_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ˇ
gradients/MatMul_grad/SumSumgradients/MatMul_grad/MatMul+gradients/MatMul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*$
_output_shapes
:

gradients/MatMul_grad/ReshapeReshapegradients/MatMul_grad/Sumgradients/MatMul_grad/Shape*
Tshape0*$
_output_shapes
:*
T0
¸
gradients/MatMul_grad/Sum_1Sumgradients/MatMul_grad/MatMul_1-gradients/MatMul_grad/BroadcastGradientArgs:1*
_output_shapes
:	*

Tidx0*
	keep_dims( *
T0

gradients/MatMul_grad/Reshape_1Reshapegradients/MatMul_grad/Sum_1gradients/MatMul_grad/Shape_1*
T0*
_output_shapes
:	*
Tshape0
p
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/Reshape ^gradients/MatMul_grad/Reshape_1
ă
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/Reshape'^gradients/MatMul_grad/tuple/group_deps*
T0*$
_output_shapes
:*0
_class&
$"loc:@gradients/MatMul_grad/Reshape
ä
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/Reshape_1'^gradients/MatMul_grad/tuple/group_deps*2
_class(
&$loc:@gradients/MatMul_grad/Reshape_1*
T0*
_output_shapes
:	
b
gradients/Reshape_grad/ShapeShapeconcat*
T0*
_output_shapes
:*
out_type0
´
gradients/Reshape_grad/ReshapeReshape.gradients/MatMul_grad/tuple/control_dependencygradients/Reshape_grad/Shape*
T0*
Tshape0*$
_output_shapes
:
\
gradients/concat_grad/RankConst*
_output_shapes
: *
value	B :*
dtype0
o
gradients/concat_grad/modFloorModconcat/axisgradients/concat_grad/Rank*
_output_shapes
: *
T0
~
gradients/concat_grad/ShapeShape#bidirectional_rnn/fw/fw/transpose_1*
_output_shapes
:*
T0*
out_type0
 
gradients/concat_grad/ShapeNShapeN#bidirectional_rnn/fw/fw/transpose_1ReverseSequence*
T0*
out_type0* 
_output_shapes
::*
N
ś
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/modgradients/concat_grad/ShapeNgradients/concat_grad/ShapeN:1*
N* 
_output_shapes
::
Ę
gradients/concat_grad/SliceSlicegradients/Reshape_grad/Reshape"gradients/concat_grad/ConcatOffsetgradients/concat_grad/ShapeN*
Index0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
T0
Đ
gradients/concat_grad/Slice_1Slicegradients/Reshape_grad/Reshape$gradients/concat_grad/ConcatOffset:1gradients/concat_grad/ShapeN:1*
T0*
Index0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@
l
&gradients/concat_grad/tuple/group_depsNoOp^gradients/concat_grad/Slice^gradients/concat_grad/Slice_1
ç
.gradients/concat_grad/tuple/control_dependencyIdentitygradients/concat_grad/Slice'^gradients/concat_grad/tuple/group_deps*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@*.
_class$
" loc:@gradients/concat_grad/Slice
í
0gradients/concat_grad/tuple/control_dependency_1Identitygradients/concat_grad/Slice_1'^gradients/concat_grad/tuple/group_deps*0
_class&
$"loc:@gradients/concat_grad/Slice_1*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
T0
 
Dgradients/bidirectional_rnn/fw/fw/transpose_1_grad/InvertPermutationInvertPermutation bidirectional_rnn/fw/fw/concat_2*
T0*
_output_shapes
:

<gradients/bidirectional_rnn/fw/fw/transpose_1_grad/transpose	Transpose.gradients/concat_grad/tuple/control_dependencyDgradients/bidirectional_rnn/fw/fw/transpose_1_grad/InvertPermutation*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
Tperm0
ĺ
.gradients/ReverseSequence_grad/ReverseSequenceReverseSequence0gradients/concat_grad/tuple/control_dependency_1Placeholder_2*
seq_dim*
	batch_dim *,
_output_shapes
:˙˙˙˙˙˙˙˙˙@*

Tlen0*
T0
ş
mgradients/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3#bidirectional_rnn/fw/fw/TensorArray$bidirectional_rnn/fw/fw/while/Exit_2*
_output_shapes

:: *
source	gradients*6
_class,
*(loc:@bidirectional_rnn/fw/fw/TensorArray
ä
igradients/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flowIdentity$bidirectional_rnn/fw/fw/while/Exit_2n^gradients/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3*6
_class,
*(loc:@bidirectional_rnn/fw/fw/TensorArray*
_output_shapes
: *
T0
ô
sgradients/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3mgradients/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3.bidirectional_rnn/fw/fw/TensorArrayStack/range<gradients/bidirectional_rnn/fw/fw/transpose_1_grad/transposeigradients/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flow*
T0*
_output_shapes
: 
p
gradients/zeros/shape_as_tensorConst*
valueB"   @   *
dtype0*
_output_shapes
:
Z
gradients/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zerosFillgradients/zeros/shape_as_tensorgradients/zeros/Const*
T0*
_output_shapes
:	@*

index_type0
r
!gradients/zeros_1/shape_as_tensorConst*
dtype0*
valueB"   @   *
_output_shapes
:
\
gradients/zeros_1/ConstConst*
valueB
 *    *
_output_shapes
: *
dtype0

gradients/zeros_1Fill!gradients/zeros_1/shape_as_tensorgradients/zeros_1/Const*
T0*

index_type0*
_output_shapes
:	@
 
Dgradients/bidirectional_rnn/bw/bw/transpose_1_grad/InvertPermutationInvertPermutation bidirectional_rnn/bw/bw/concat_2*
_output_shapes
:*
T0

<gradients/bidirectional_rnn/bw/bw/transpose_1_grad/transpose	Transpose.gradients/ReverseSequence_grad/ReverseSequenceDgradients/bidirectional_rnn/bw/bw/transpose_1_grad/InvertPermutation*
Tperm0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@
Î
:gradients/bidirectional_rnn/fw/fw/while/Exit_2_grad/b_exitEntersgradients/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3*E

frame_name75gradients/bidirectional_rnn/fw/fw/while/while_context*
parallel_iterations *
_output_shapes
: *
is_constant( *
T0
ó
:gradients/bidirectional_rnn/fw/fw/while/Exit_3_grad/b_exitEntergradients/zeros*
_output_shapes
:	@*E

frame_name75gradients/bidirectional_rnn/fw/fw/while/while_context*
is_constant( *
T0*
parallel_iterations 
ő
:gradients/bidirectional_rnn/fw/fw/while/Exit_4_grad/b_exitEntergradients/zeros_1*
T0*
parallel_iterations *
_output_shapes
:	@*
is_constant( *E

frame_name75gradients/bidirectional_rnn/fw/fw/while/while_context
ş
mgradients/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3#bidirectional_rnn/bw/bw/TensorArray$bidirectional_rnn/bw/bw/while/Exit_2*6
_class,
*(loc:@bidirectional_rnn/bw/bw/TensorArray*
source	gradients*
_output_shapes

:: 
ä
igradients/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flowIdentity$bidirectional_rnn/bw/bw/while/Exit_2n^gradients/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3*
_output_shapes
: *6
_class,
*(loc:@bidirectional_rnn/bw/bw/TensorArray*
T0
ô
sgradients/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3mgradients/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3.bidirectional_rnn/bw/bw/TensorArrayStack/range<gradients/bidirectional_rnn/bw/bw/transpose_1_grad/transposeigradients/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flow*
T0*
_output_shapes
: 
r
!gradients/zeros_2/shape_as_tensorConst*
dtype0*
valueB"   @   *
_output_shapes
:
\
gradients/zeros_2/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_2Fill!gradients/zeros_2/shape_as_tensorgradients/zeros_2/Const*

index_type0*
T0*
_output_shapes
:	@
r
!gradients/zeros_3/shape_as_tensorConst*
dtype0*
valueB"   @   *
_output_shapes
:
\
gradients/zeros_3/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_3Fill!gradients/zeros_3/shape_as_tensorgradients/zeros_3/Const*

index_type0*
_output_shapes
:	@*
T0
ö
>gradients/bidirectional_rnn/fw/fw/while/Switch_2_grad/b_switchMerge:gradients/bidirectional_rnn/fw/fw/while/Exit_2_grad/b_exitEgradients/bidirectional_rnn/fw/fw/while/Switch_2_grad_1/NextIteration*
T0*
N*
_output_shapes
: : 
˙
>gradients/bidirectional_rnn/fw/fw/while/Switch_3_grad/b_switchMerge:gradients/bidirectional_rnn/fw/fw/while/Exit_3_grad/b_exitEgradients/bidirectional_rnn/fw/fw/while/Switch_3_grad_1/NextIteration*
T0*!
_output_shapes
:	@: *
N
˙
>gradients/bidirectional_rnn/fw/fw/while/Switch_4_grad/b_switchMerge:gradients/bidirectional_rnn/fw/fw/while/Exit_4_grad/b_exitEgradients/bidirectional_rnn/fw/fw/while/Switch_4_grad_1/NextIteration*
N*
T0*!
_output_shapes
:	@: 
Î
:gradients/bidirectional_rnn/bw/bw/while/Exit_2_grad/b_exitEntersgradients/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3*
parallel_iterations *E

frame_name75gradients/bidirectional_rnn/bw/bw/while/while_context*
is_constant( *
T0*
_output_shapes
: 
ő
:gradients/bidirectional_rnn/bw/bw/while/Exit_3_grad/b_exitEntergradients/zeros_2*
parallel_iterations *
T0*
is_constant( *E

frame_name75gradients/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:	@
ő
:gradients/bidirectional_rnn/bw/bw/while/Exit_4_grad/b_exitEntergradients/zeros_3*
parallel_iterations *E

frame_name75gradients/bidirectional_rnn/bw/bw/while/while_context*
T0*
is_constant( *
_output_shapes
:	@

;gradients/bidirectional_rnn/fw/fw/while/Merge_2_grad/SwitchSwitch>gradients/bidirectional_rnn/fw/fw/while/Switch_2_grad/b_switchgradients/b_count_2*
_output_shapes
: : *
T0*Q
_classG
ECloc:@gradients/bidirectional_rnn/fw/fw/while/Switch_2_grad/b_switch

Egradients/bidirectional_rnn/fw/fw/while/Merge_2_grad/tuple/group_depsNoOp<^gradients/bidirectional_rnn/fw/fw/while/Merge_2_grad/Switch
Ň
Mgradients/bidirectional_rnn/fw/fw/while/Merge_2_grad/tuple/control_dependencyIdentity;gradients/bidirectional_rnn/fw/fw/while/Merge_2_grad/SwitchF^gradients/bidirectional_rnn/fw/fw/while/Merge_2_grad/tuple/group_deps*Q
_classG
ECloc:@gradients/bidirectional_rnn/fw/fw/while/Switch_2_grad/b_switch*
T0*
_output_shapes
: 
Ö
Ogradients/bidirectional_rnn/fw/fw/while/Merge_2_grad/tuple/control_dependency_1Identity=gradients/bidirectional_rnn/fw/fw/while/Merge_2_grad/Switch:1F^gradients/bidirectional_rnn/fw/fw/while/Merge_2_grad/tuple/group_deps*
_output_shapes
: *Q
_classG
ECloc:@gradients/bidirectional_rnn/fw/fw/while/Switch_2_grad/b_switch*
T0
˘
;gradients/bidirectional_rnn/fw/fw/while/Merge_3_grad/SwitchSwitch>gradients/bidirectional_rnn/fw/fw/while/Switch_3_grad/b_switchgradients/b_count_2*Q
_classG
ECloc:@gradients/bidirectional_rnn/fw/fw/while/Switch_3_grad/b_switch*
T0**
_output_shapes
:	@:	@

Egradients/bidirectional_rnn/fw/fw/while/Merge_3_grad/tuple/group_depsNoOp<^gradients/bidirectional_rnn/fw/fw/while/Merge_3_grad/Switch
Ű
Mgradients/bidirectional_rnn/fw/fw/while/Merge_3_grad/tuple/control_dependencyIdentity;gradients/bidirectional_rnn/fw/fw/while/Merge_3_grad/SwitchF^gradients/bidirectional_rnn/fw/fw/while/Merge_3_grad/tuple/group_deps*
_output_shapes
:	@*Q
_classG
ECloc:@gradients/bidirectional_rnn/fw/fw/while/Switch_3_grad/b_switch*
T0
ß
Ogradients/bidirectional_rnn/fw/fw/while/Merge_3_grad/tuple/control_dependency_1Identity=gradients/bidirectional_rnn/fw/fw/while/Merge_3_grad/Switch:1F^gradients/bidirectional_rnn/fw/fw/while/Merge_3_grad/tuple/group_deps*
T0*
_output_shapes
:	@*Q
_classG
ECloc:@gradients/bidirectional_rnn/fw/fw/while/Switch_3_grad/b_switch
˘
;gradients/bidirectional_rnn/fw/fw/while/Merge_4_grad/SwitchSwitch>gradients/bidirectional_rnn/fw/fw/while/Switch_4_grad/b_switchgradients/b_count_2**
_output_shapes
:	@:	@*Q
_classG
ECloc:@gradients/bidirectional_rnn/fw/fw/while/Switch_4_grad/b_switch*
T0

Egradients/bidirectional_rnn/fw/fw/while/Merge_4_grad/tuple/group_depsNoOp<^gradients/bidirectional_rnn/fw/fw/while/Merge_4_grad/Switch
Ű
Mgradients/bidirectional_rnn/fw/fw/while/Merge_4_grad/tuple/control_dependencyIdentity;gradients/bidirectional_rnn/fw/fw/while/Merge_4_grad/SwitchF^gradients/bidirectional_rnn/fw/fw/while/Merge_4_grad/tuple/group_deps*
T0*
_output_shapes
:	@*Q
_classG
ECloc:@gradients/bidirectional_rnn/fw/fw/while/Switch_4_grad/b_switch
ß
Ogradients/bidirectional_rnn/fw/fw/while/Merge_4_grad/tuple/control_dependency_1Identity=gradients/bidirectional_rnn/fw/fw/while/Merge_4_grad/Switch:1F^gradients/bidirectional_rnn/fw/fw/while/Merge_4_grad/tuple/group_deps*Q
_classG
ECloc:@gradients/bidirectional_rnn/fw/fw/while/Switch_4_grad/b_switch*
T0*
_output_shapes
:	@
ö
>gradients/bidirectional_rnn/bw/bw/while/Switch_2_grad/b_switchMerge:gradients/bidirectional_rnn/bw/bw/while/Exit_2_grad/b_exitEgradients/bidirectional_rnn/bw/bw/while/Switch_2_grad_1/NextIteration*
T0*
_output_shapes
: : *
N
˙
>gradients/bidirectional_rnn/bw/bw/while/Switch_3_grad/b_switchMerge:gradients/bidirectional_rnn/bw/bw/while/Exit_3_grad/b_exitEgradients/bidirectional_rnn/bw/bw/while/Switch_3_grad_1/NextIteration*!
_output_shapes
:	@: *
N*
T0
˙
>gradients/bidirectional_rnn/bw/bw/while/Switch_4_grad/b_switchMerge:gradients/bidirectional_rnn/bw/bw/while/Exit_4_grad/b_exitEgradients/bidirectional_rnn/bw/bw/while/Switch_4_grad_1/NextIteration*
T0*!
_output_shapes
:	@: *
N
ą
9gradients/bidirectional_rnn/fw/fw/while/Enter_2_grad/ExitExitMgradients/bidirectional_rnn/fw/fw/while/Merge_2_grad/tuple/control_dependency*
_output_shapes
: *
T0
ş
9gradients/bidirectional_rnn/fw/fw/while/Enter_3_grad/ExitExitMgradients/bidirectional_rnn/fw/fw/while/Merge_3_grad/tuple/control_dependency*
_output_shapes
:	@*
T0
ş
9gradients/bidirectional_rnn/fw/fw/while/Enter_4_grad/ExitExitMgradients/bidirectional_rnn/fw/fw/while/Merge_4_grad/tuple/control_dependency*
_output_shapes
:	@*
T0

;gradients/bidirectional_rnn/bw/bw/while/Merge_2_grad/SwitchSwitch>gradients/bidirectional_rnn/bw/bw/while/Switch_2_grad/b_switchgradients/b_count_6*Q
_classG
ECloc:@gradients/bidirectional_rnn/bw/bw/while/Switch_2_grad/b_switch*
_output_shapes
: : *
T0

Egradients/bidirectional_rnn/bw/bw/while/Merge_2_grad/tuple/group_depsNoOp<^gradients/bidirectional_rnn/bw/bw/while/Merge_2_grad/Switch
Ň
Mgradients/bidirectional_rnn/bw/bw/while/Merge_2_grad/tuple/control_dependencyIdentity;gradients/bidirectional_rnn/bw/bw/while/Merge_2_grad/SwitchF^gradients/bidirectional_rnn/bw/bw/while/Merge_2_grad/tuple/group_deps*
_output_shapes
: *Q
_classG
ECloc:@gradients/bidirectional_rnn/bw/bw/while/Switch_2_grad/b_switch*
T0
Ö
Ogradients/bidirectional_rnn/bw/bw/while/Merge_2_grad/tuple/control_dependency_1Identity=gradients/bidirectional_rnn/bw/bw/while/Merge_2_grad/Switch:1F^gradients/bidirectional_rnn/bw/bw/while/Merge_2_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/bidirectional_rnn/bw/bw/while/Switch_2_grad/b_switch*
_output_shapes
: 
˘
;gradients/bidirectional_rnn/bw/bw/while/Merge_3_grad/SwitchSwitch>gradients/bidirectional_rnn/bw/bw/while/Switch_3_grad/b_switchgradients/b_count_6*
T0*Q
_classG
ECloc:@gradients/bidirectional_rnn/bw/bw/while/Switch_3_grad/b_switch**
_output_shapes
:	@:	@

Egradients/bidirectional_rnn/bw/bw/while/Merge_3_grad/tuple/group_depsNoOp<^gradients/bidirectional_rnn/bw/bw/while/Merge_3_grad/Switch
Ű
Mgradients/bidirectional_rnn/bw/bw/while/Merge_3_grad/tuple/control_dependencyIdentity;gradients/bidirectional_rnn/bw/bw/while/Merge_3_grad/SwitchF^gradients/bidirectional_rnn/bw/bw/while/Merge_3_grad/tuple/group_deps*Q
_classG
ECloc:@gradients/bidirectional_rnn/bw/bw/while/Switch_3_grad/b_switch*
T0*
_output_shapes
:	@
ß
Ogradients/bidirectional_rnn/bw/bw/while/Merge_3_grad/tuple/control_dependency_1Identity=gradients/bidirectional_rnn/bw/bw/while/Merge_3_grad/Switch:1F^gradients/bidirectional_rnn/bw/bw/while/Merge_3_grad/tuple/group_deps*Q
_classG
ECloc:@gradients/bidirectional_rnn/bw/bw/while/Switch_3_grad/b_switch*
_output_shapes
:	@*
T0
˘
;gradients/bidirectional_rnn/bw/bw/while/Merge_4_grad/SwitchSwitch>gradients/bidirectional_rnn/bw/bw/while/Switch_4_grad/b_switchgradients/b_count_6*
T0*Q
_classG
ECloc:@gradients/bidirectional_rnn/bw/bw/while/Switch_4_grad/b_switch**
_output_shapes
:	@:	@

Egradients/bidirectional_rnn/bw/bw/while/Merge_4_grad/tuple/group_depsNoOp<^gradients/bidirectional_rnn/bw/bw/while/Merge_4_grad/Switch
Ű
Mgradients/bidirectional_rnn/bw/bw/while/Merge_4_grad/tuple/control_dependencyIdentity;gradients/bidirectional_rnn/bw/bw/while/Merge_4_grad/SwitchF^gradients/bidirectional_rnn/bw/bw/while/Merge_4_grad/tuple/group_deps*
_output_shapes
:	@*
T0*Q
_classG
ECloc:@gradients/bidirectional_rnn/bw/bw/while/Switch_4_grad/b_switch
ß
Ogradients/bidirectional_rnn/bw/bw/while/Merge_4_grad/tuple/control_dependency_1Identity=gradients/bidirectional_rnn/bw/bw/while/Merge_4_grad/Switch:1F^gradients/bidirectional_rnn/bw/bw/while/Merge_4_grad/tuple/group_deps*
_output_shapes
:	@*Q
_classG
ECloc:@gradients/bidirectional_rnn/bw/bw/while/Switch_4_grad/b_switch*
T0
Ç
rgradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3xgradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterOgradients/bidirectional_rnn/fw/fw/while/Merge_2_grad/tuple/control_dependency_1*
source	gradients*
_output_shapes

:: *>
_class4
20loc:@bidirectional_rnn/fw/fw/while/dropout/mul_1

xgradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnter#bidirectional_rnn/fw/fw/TensorArray*E

frame_name75gradients/bidirectional_rnn/fw/fw/while/while_context*
T0*
parallel_iterations *
is_constant(*
_output_shapes
:*>
_class4
20loc:@bidirectional_rnn/fw/fw/while/dropout/mul_1
Ą
ngradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flowIdentityOgradients/bidirectional_rnn/fw/fw/while/Merge_2_grad/tuple/control_dependency_1s^gradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3*>
_class4
20loc:@bidirectional_rnn/fw/fw/while/dropout/mul_1*
_output_shapes
: *
T0
ř
bgradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3TensorArrayReadV3rgradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3mgradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2ngradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flow*
_output_shapes
:	@*
dtype0
đ
hgradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/ConstConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: *;
_class1
/-loc:@bidirectional_rnn/fw/fw/while/Identity_1
Ů
hgradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_accStackV2hgradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Const*
	elem_type0*;
_class1
/-loc:@bidirectional_rnn/fw/fw/while/Identity_1*

stack_name *
_output_shapes
:
ë
hgradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/EnterEnterhgradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
is_constant(*
_output_shapes
:*
T0*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context*
parallel_iterations 
Ő
ngradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2StackPushV2hgradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter(bidirectional_rnn/fw/fw/while/Identity_1^gradients/Add*
T0*
swap_memory( *
_output_shapes
: 
Š
mgradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
StackPopV2sgradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/Enter^gradients/Sub*
_output_shapes
: *
	elem_type0

sgradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/EnterEnterhgradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
is_constant(*E

frame_name75gradients/bidirectional_rnn/fw/fw/while/while_context*
T0*
_output_shapes
:*
parallel_iterations 
š	
igradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_syncControlTriggerH^gradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2n^gradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2X^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2R^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2T^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2R^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2T^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2P^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2R^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2J^gradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul/StackPopV2L^gradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul_1/StackPopV2H^gradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/StackPopV2J^gradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/StackPopV2
 
agradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_depsNoOpP^gradients/bidirectional_rnn/fw/fw/while/Merge_2_grad/tuple/control_dependency_1c^gradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3
Ţ
igradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependencyIdentitybgradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3b^gradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*
_output_shapes
:	@*u
_classk
igloc:@gradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3
 
kgradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1IdentityOgradients/bidirectional_rnn/fw/fw/while/Merge_2_grad/tuple/control_dependency_1b^gradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/bidirectional_rnn/fw/fw/while/Switch_2_grad/b_switch*
_output_shapes
: 
ą
Pgradients/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/shape_as_tensorConst^gradients/Sub*
valueB"   @   *
_output_shapes
:*
dtype0

Fgradients/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/ConstConst^gradients/Sub*
valueB
 *    *
_output_shapes
: *
dtype0

@gradients/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_likeFillPgradients/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/shape_as_tensorFgradients/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/Const*
_output_shapes
:	@*
T0*

index_type0
Ě
<gradients/bidirectional_rnn/fw/fw/while/Select_1_grad/SelectSelectGgradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2Ogradients/bidirectional_rnn/fw/fw/while/Merge_3_grad/tuple/control_dependency_1@gradients/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like*
T0*
_output_shapes
:	@
Ě
Bgradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/ConstConst*
dtype0*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙*=
_class3
1/loc:@bidirectional_rnn/fw/fw/while/GreaterEqual

Bgradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/f_accStackV2Bgradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/Const*

stack_name *=
_class3
1/loc:@bidirectional_rnn/fw/fw/while/GreaterEqual*
	elem_type0
*
_output_shapes
:

Bgradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/EnterEnterBgradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/f_acc*
parallel_iterations *
is_constant(*
T0*
_output_shapes
:*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context

Hgradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPushV2StackPushV2Bgradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/Enter*bidirectional_rnn/fw/fw/while/GreaterEqual^gradients/Add*
T0
*
_output_shapes
:*
swap_memory( 
ß
Ggradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2
StackPopV2Mgradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2/Enter^gradients/Sub*
_output_shapes
:*
	elem_type0

´
Mgradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2/EnterEnterBgradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/f_acc*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations *E

frame_name75gradients/bidirectional_rnn/fw/fw/while/while_context
Î
>gradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select_1SelectGgradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2@gradients/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_likeOgradients/bidirectional_rnn/fw/fw/while/Merge_3_grad/tuple/control_dependency_1*
T0*
_output_shapes
:	@
Î
Fgradients/bidirectional_rnn/fw/fw/while/Select_1_grad/tuple/group_depsNoOp=^gradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select?^gradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select_1
Ü
Ngradients/bidirectional_rnn/fw/fw/while/Select_1_grad/tuple/control_dependencyIdentity<gradients/bidirectional_rnn/fw/fw/while/Select_1_grad/SelectG^gradients/bidirectional_rnn/fw/fw/while/Select_1_grad/tuple/group_deps*
T0*
_output_shapes
:	@*O
_classE
CAloc:@gradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select
â
Pgradients/bidirectional_rnn/fw/fw/while/Select_1_grad/tuple/control_dependency_1Identity>gradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select_1G^gradients/bidirectional_rnn/fw/fw/while/Select_1_grad/tuple/group_deps*
T0*
_output_shapes
:	@*Q
_classG
ECloc:@gradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select_1
ą
Pgradients/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/shape_as_tensorConst^gradients/Sub*
_output_shapes
:*
dtype0*
valueB"   @   

Fgradients/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/ConstConst^gradients/Sub*
dtype0*
valueB
 *    *
_output_shapes
: 

@gradients/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_likeFillPgradients/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/shape_as_tensorFgradients/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/Const*

index_type0*
T0*
_output_shapes
:	@
Ě
<gradients/bidirectional_rnn/fw/fw/while/Select_2_grad/SelectSelectGgradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2Ogradients/bidirectional_rnn/fw/fw/while/Merge_4_grad/tuple/control_dependency_1@gradients/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like*
_output_shapes
:	@*
T0
Î
>gradients/bidirectional_rnn/fw/fw/while/Select_2_grad/Select_1SelectGgradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2@gradients/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_likeOgradients/bidirectional_rnn/fw/fw/while/Merge_4_grad/tuple/control_dependency_1*
_output_shapes
:	@*
T0
Î
Fgradients/bidirectional_rnn/fw/fw/while/Select_2_grad/tuple/group_depsNoOp=^gradients/bidirectional_rnn/fw/fw/while/Select_2_grad/Select?^gradients/bidirectional_rnn/fw/fw/while/Select_2_grad/Select_1
Ü
Ngradients/bidirectional_rnn/fw/fw/while/Select_2_grad/tuple/control_dependencyIdentity<gradients/bidirectional_rnn/fw/fw/while/Select_2_grad/SelectG^gradients/bidirectional_rnn/fw/fw/while/Select_2_grad/tuple/group_deps*
_output_shapes
:	@*O
_classE
CAloc:@gradients/bidirectional_rnn/fw/fw/while/Select_2_grad/Select*
T0
â
Pgradients/bidirectional_rnn/fw/fw/while/Select_2_grad/tuple/control_dependency_1Identity>gradients/bidirectional_rnn/fw/fw/while/Select_2_grad/Select_1G^gradients/bidirectional_rnn/fw/fw/while/Select_2_grad/tuple/group_deps*Q
_classG
ECloc:@gradients/bidirectional_rnn/fw/fw/while/Select_2_grad/Select_1*
_output_shapes
:	@*
T0
ą
9gradients/bidirectional_rnn/bw/bw/while/Enter_2_grad/ExitExitMgradients/bidirectional_rnn/bw/bw/while/Merge_2_grad/tuple/control_dependency*
T0*
_output_shapes
: 
ş
9gradients/bidirectional_rnn/bw/bw/while/Enter_3_grad/ExitExitMgradients/bidirectional_rnn/bw/bw/while/Merge_3_grad/tuple/control_dependency*
_output_shapes
:	@*
T0
ş
9gradients/bidirectional_rnn/bw/bw/while/Enter_4_grad/ExitExitMgradients/bidirectional_rnn/bw/bw/while/Merge_4_grad/tuple/control_dependency*
_output_shapes
:	@*
T0
Ż
Ngradients/bidirectional_rnn/fw/fw/while/Select_grad/zeros_like/shape_as_tensorConst^gradients/Sub*
valueB"   @   *
_output_shapes
:*
dtype0

Dgradients/bidirectional_rnn/fw/fw/while/Select_grad/zeros_like/ConstConst^gradients/Sub*
_output_shapes
: *
valueB
 *    *
dtype0

>gradients/bidirectional_rnn/fw/fw/while/Select_grad/zeros_likeFillNgradients/bidirectional_rnn/fw/fw/while/Select_grad/zeros_like/shape_as_tensorDgradients/bidirectional_rnn/fw/fw/while/Select_grad/zeros_like/Const*

index_type0*
_output_shapes
:	@*
T0
â
:gradients/bidirectional_rnn/fw/fw/while/Select_grad/SelectSelectGgradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2igradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency>gradients/bidirectional_rnn/fw/fw/while/Select_grad/zeros_like*
T0*
_output_shapes
:	@
ä
<gradients/bidirectional_rnn/fw/fw/while/Select_grad/Select_1SelectGgradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2>gradients/bidirectional_rnn/fw/fw/while/Select_grad/zeros_likeigradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency*
T0*
_output_shapes
:	@
Č
Dgradients/bidirectional_rnn/fw/fw/while/Select_grad/tuple/group_depsNoOp;^gradients/bidirectional_rnn/fw/fw/while/Select_grad/Select=^gradients/bidirectional_rnn/fw/fw/while/Select_grad/Select_1
Ô
Lgradients/bidirectional_rnn/fw/fw/while/Select_grad/tuple/control_dependencyIdentity:gradients/bidirectional_rnn/fw/fw/while/Select_grad/SelectE^gradients/bidirectional_rnn/fw/fw/while/Select_grad/tuple/group_deps*
_output_shapes
:	@*M
_classC
A?loc:@gradients/bidirectional_rnn/fw/fw/while/Select_grad/Select*
T0
Ú
Ngradients/bidirectional_rnn/fw/fw/while/Select_grad/tuple/control_dependency_1Identity<gradients/bidirectional_rnn/fw/fw/while/Select_grad/Select_1E^gradients/bidirectional_rnn/fw/fw/while/Select_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/bidirectional_rnn/fw/fw/while/Select_grad/Select_1*
_output_shapes
:	@
Ç
rgradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3xgradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterOgradients/bidirectional_rnn/bw/bw/while/Merge_2_grad/tuple/control_dependency_1*>
_class4
20loc:@bidirectional_rnn/bw/bw/while/dropout/mul_1*
source	gradients*
_output_shapes

:: 

xgradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnter#bidirectional_rnn/bw/bw/TensorArray*E

frame_name75gradients/bidirectional_rnn/bw/bw/while/while_context*>
_class4
20loc:@bidirectional_rnn/bw/bw/while/dropout/mul_1*
parallel_iterations *
is_constant(*
T0*
_output_shapes
:
Ą
ngradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flowIdentityOgradients/bidirectional_rnn/bw/bw/while/Merge_2_grad/tuple/control_dependency_1s^gradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3*
_output_shapes
: *
T0*>
_class4
20loc:@bidirectional_rnn/bw/bw/while/dropout/mul_1
ř
bgradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3TensorArrayReadV3rgradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3mgradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2ngradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flow*
dtype0*
_output_shapes
:	@
đ
hgradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/ConstConst*
valueB :
˙˙˙˙˙˙˙˙˙*;
_class1
/-loc:@bidirectional_rnn/bw/bw/while/Identity_1*
dtype0*
_output_shapes
: 
Ů
hgradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_accStackV2hgradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Const*;
_class1
/-loc:@bidirectional_rnn/bw/bw/while/Identity_1*

stack_name *
_output_shapes
:*
	elem_type0
ë
hgradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/EnterEnterhgradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
_output_shapes
:*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
parallel_iterations 
×
ngradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2StackPushV2hgradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter(bidirectional_rnn/bw/bw/while/Identity_1^gradients/Add_1*
_output_shapes
: *
T0*
swap_memory( 
Ť
mgradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
StackPopV2sgradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/Enter^gradients/Sub_1*
_output_shapes
: *
	elem_type0

sgradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/EnterEnterhgradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*E

frame_name75gradients/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:*
parallel_iterations *
is_constant(*
T0
š	
igradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_syncControlTriggerH^gradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2n^gradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2X^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2R^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2T^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2R^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2T^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2P^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2R^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2J^gradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul/StackPopV2L^gradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul_1/StackPopV2H^gradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/StackPopV2J^gradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/StackPopV2
 
agradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_depsNoOpP^gradients/bidirectional_rnn/bw/bw/while/Merge_2_grad/tuple/control_dependency_1c^gradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3
Ţ
igradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependencyIdentitybgradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3b^gradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*u
_classk
igloc:@gradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3*
_output_shapes
:	@
 
kgradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1IdentityOgradients/bidirectional_rnn/bw/bw/while/Merge_2_grad/tuple/control_dependency_1b^gradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*
_output_shapes
: *Q
_classG
ECloc:@gradients/bidirectional_rnn/bw/bw/while/Switch_2_grad/b_switch
ł
Pgradients/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/shape_as_tensorConst^gradients/Sub_1*
valueB"   @   *
dtype0*
_output_shapes
:

Fgradients/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/ConstConst^gradients/Sub_1*
_output_shapes
: *
dtype0*
valueB
 *    

@gradients/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_likeFillPgradients/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/shape_as_tensorFgradients/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/Const*
_output_shapes
:	@*

index_type0*
T0
Ě
<gradients/bidirectional_rnn/bw/bw/while/Select_1_grad/SelectSelectGgradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2Ogradients/bidirectional_rnn/bw/bw/while/Merge_3_grad/tuple/control_dependency_1@gradients/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like*
T0*
_output_shapes
:	@
Ě
Bgradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/ConstConst*
dtype0*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙*=
_class3
1/loc:@bidirectional_rnn/bw/bw/while/GreaterEqual

Bgradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/f_accStackV2Bgradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/Const*
	elem_type0
*=
_class3
1/loc:@bidirectional_rnn/bw/bw/while/GreaterEqual*

stack_name *
_output_shapes
:

Bgradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/EnterEnterBgradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/f_acc*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:*
is_constant(*
T0*
parallel_iterations 

Hgradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPushV2StackPushV2Bgradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/Enter*bidirectional_rnn/bw/bw/while/GreaterEqual^gradients/Add_1*
swap_memory( *
_output_shapes
:*
T0

á
Ggradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2
StackPopV2Mgradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2/Enter^gradients/Sub_1*
	elem_type0
*
_output_shapes
:
´
Mgradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2/EnterEnterBgradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/f_acc*
parallel_iterations *
_output_shapes
:*
is_constant(*
T0*E

frame_name75gradients/bidirectional_rnn/bw/bw/while/while_context
Î
>gradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select_1SelectGgradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2@gradients/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_likeOgradients/bidirectional_rnn/bw/bw/while/Merge_3_grad/tuple/control_dependency_1*
T0*
_output_shapes
:	@
Î
Fgradients/bidirectional_rnn/bw/bw/while/Select_1_grad/tuple/group_depsNoOp=^gradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select?^gradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select_1
Ü
Ngradients/bidirectional_rnn/bw/bw/while/Select_1_grad/tuple/control_dependencyIdentity<gradients/bidirectional_rnn/bw/bw/while/Select_1_grad/SelectG^gradients/bidirectional_rnn/bw/bw/while/Select_1_grad/tuple/group_deps*
_output_shapes
:	@*
T0*O
_classE
CAloc:@gradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select
â
Pgradients/bidirectional_rnn/bw/bw/while/Select_1_grad/tuple/control_dependency_1Identity>gradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select_1G^gradients/bidirectional_rnn/bw/bw/while/Select_1_grad/tuple/group_deps*
_output_shapes
:	@*Q
_classG
ECloc:@gradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select_1*
T0
ł
Pgradients/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/shape_as_tensorConst^gradients/Sub_1*
_output_shapes
:*
dtype0*
valueB"   @   

Fgradients/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/ConstConst^gradients/Sub_1*
_output_shapes
: *
valueB
 *    *
dtype0

@gradients/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_likeFillPgradients/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/shape_as_tensorFgradients/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/Const*
T0*
_output_shapes
:	@*

index_type0
Ě
<gradients/bidirectional_rnn/bw/bw/while/Select_2_grad/SelectSelectGgradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2Ogradients/bidirectional_rnn/bw/bw/while/Merge_4_grad/tuple/control_dependency_1@gradients/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like*
T0*
_output_shapes
:	@
Î
>gradients/bidirectional_rnn/bw/bw/while/Select_2_grad/Select_1SelectGgradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2@gradients/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_likeOgradients/bidirectional_rnn/bw/bw/while/Merge_4_grad/tuple/control_dependency_1*
_output_shapes
:	@*
T0
Î
Fgradients/bidirectional_rnn/bw/bw/while/Select_2_grad/tuple/group_depsNoOp=^gradients/bidirectional_rnn/bw/bw/while/Select_2_grad/Select?^gradients/bidirectional_rnn/bw/bw/while/Select_2_grad/Select_1
Ü
Ngradients/bidirectional_rnn/bw/bw/while/Select_2_grad/tuple/control_dependencyIdentity<gradients/bidirectional_rnn/bw/bw/while/Select_2_grad/SelectG^gradients/bidirectional_rnn/bw/bw/while/Select_2_grad/tuple/group_deps*O
_classE
CAloc:@gradients/bidirectional_rnn/bw/bw/while/Select_2_grad/Select*
_output_shapes
:	@*
T0
â
Pgradients/bidirectional_rnn/bw/bw/while/Select_2_grad/tuple/control_dependency_1Identity>gradients/bidirectional_rnn/bw/bw/while/Select_2_grad/Select_1G^gradients/bidirectional_rnn/bw/bw/while/Select_2_grad/tuple/group_deps*
_output_shapes
:	@*
T0*Q
_classG
ECloc:@gradients/bidirectional_rnn/bw/bw/while/Select_2_grad/Select_1

>gradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/MulMulNgradients/bidirectional_rnn/fw/fw/while/Select_grad/tuple/control_dependency_1Igradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul/StackPopV2*
_output_shapes
:	@*
T0
Î
Dgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul/ConstConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*=
_class3
1/loc:@bidirectional_rnn/fw/fw/while/dropout/Cast*
_output_shapes
: 

Dgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul/f_accStackV2Dgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul/Const*

stack_name *
_output_shapes
:*=
_class3
1/loc:@bidirectional_rnn/fw/fw/while/dropout/Cast*
	elem_type0
Ł
Dgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul/EnterEnterDgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul/f_acc*
T0*
parallel_iterations *;

frame_name-+bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:*
is_constant(

Jgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul/StackPushV2StackPushV2Dgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul/Enter*bidirectional_rnn/fw/fw/while/dropout/Cast^gradients/Add*
_output_shapes
:	@*
swap_memory( *
T0
ę
Igradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul/StackPopV2
StackPopV2Ogradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:	@
¸
Ogradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul/StackPopV2/EnterEnterDgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul/f_acc*
T0*
parallel_iterations *
is_constant(*E

frame_name75gradients/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:

@gradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul_1MulNgradients/bidirectional_rnn/fw/fw/while/Select_grad/tuple/control_dependency_1Kgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul_1/StackPopV2*
T0*
_output_shapes
:	@
Ď
Fgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul_1/ConstConst*
dtype0*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙*<
_class2
0.loc:@bidirectional_rnn/fw/fw/while/dropout/mul

Fgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul_1/f_accStackV2Fgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul_1/Const*

stack_name *<
_class2
0.loc:@bidirectional_rnn/fw/fw/while/dropout/mul*
_output_shapes
:*
	elem_type0
§
Fgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul_1/EnterEnterFgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul_1/f_acc*
_output_shapes
:*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context*
parallel_iterations *
is_constant(*
T0

Lgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul_1/StackPushV2StackPushV2Fgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul_1/Enter)bidirectional_rnn/fw/fw/while/dropout/mul^gradients/Add*
swap_memory( *
T0*
_output_shapes
:	@
î
Kgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul_1/StackPopV2
StackPopV2Qgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:	@
ź
Qgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul_1/StackPopV2/EnterEnterFgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*E

frame_name75gradients/bidirectional_rnn/fw/fw/while/while_context*
parallel_iterations *
_output_shapes
:
×
Kgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/tuple/group_depsNoOp?^gradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/MulA^gradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul_1
ę
Sgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/tuple/control_dependencyIdentity>gradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/MulL^gradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/tuple/group_deps*
T0*
_output_shapes
:	@*Q
_classG
ECloc:@gradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul
đ
Ugradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/tuple/control_dependency_1Identity@gradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul_1L^gradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/tuple/group_deps*S
_classI
GEloc:@gradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul_1*
_output_shapes
:	@*
T0
ä
Egradients/bidirectional_rnn/fw/fw/while/Switch_2_grad_1/NextIterationNextIterationkgradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1*
_output_shapes
: *
T0
ą
Ngradients/bidirectional_rnn/bw/bw/while/Select_grad/zeros_like/shape_as_tensorConst^gradients/Sub_1*
dtype0*
_output_shapes
:*
valueB"   @   

Dgradients/bidirectional_rnn/bw/bw/while/Select_grad/zeros_like/ConstConst^gradients/Sub_1*
_output_shapes
: *
valueB
 *    *
dtype0

>gradients/bidirectional_rnn/bw/bw/while/Select_grad/zeros_likeFillNgradients/bidirectional_rnn/bw/bw/while/Select_grad/zeros_like/shape_as_tensorDgradients/bidirectional_rnn/bw/bw/while/Select_grad/zeros_like/Const*
_output_shapes
:	@*

index_type0*
T0
â
:gradients/bidirectional_rnn/bw/bw/while/Select_grad/SelectSelectGgradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2igradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency>gradients/bidirectional_rnn/bw/bw/while/Select_grad/zeros_like*
_output_shapes
:	@*
T0
ä
<gradients/bidirectional_rnn/bw/bw/while/Select_grad/Select_1SelectGgradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2>gradients/bidirectional_rnn/bw/bw/while/Select_grad/zeros_likeigradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency*
T0*
_output_shapes
:	@
Č
Dgradients/bidirectional_rnn/bw/bw/while/Select_grad/tuple/group_depsNoOp;^gradients/bidirectional_rnn/bw/bw/while/Select_grad/Select=^gradients/bidirectional_rnn/bw/bw/while/Select_grad/Select_1
Ô
Lgradients/bidirectional_rnn/bw/bw/while/Select_grad/tuple/control_dependencyIdentity:gradients/bidirectional_rnn/bw/bw/while/Select_grad/SelectE^gradients/bidirectional_rnn/bw/bw/while/Select_grad/tuple/group_deps*
T0*
_output_shapes
:	@*M
_classC
A?loc:@gradients/bidirectional_rnn/bw/bw/while/Select_grad/Select
Ú
Ngradients/bidirectional_rnn/bw/bw/while/Select_grad/tuple/control_dependency_1Identity<gradients/bidirectional_rnn/bw/bw/while/Select_grad/Select_1E^gradients/bidirectional_rnn/bw/bw/while/Select_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/bidirectional_rnn/bw/bw/while/Select_grad/Select_1*
_output_shapes
:	@

>gradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/ShapeConst^gradients/Sub*
valueB"   @   *
dtype0*
_output_shapes
:

@gradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Shape_1Const^gradients/Sub*
_output_shapes
: *
dtype0*
valueB 
Ś
Ngradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs>gradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Shape@gradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

<gradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/MulMulSgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/tuple/control_dependencyGgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/StackPopV2*
_output_shapes
:	@*
T0
Ď
Bgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/ConstConst*
_output_shapes
: *@
_class6
42loc:@bidirectional_rnn/fw/fw/while/dropout/truediv*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0

Bgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/f_accStackV2Bgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/Const*
_output_shapes
:*

stack_name *
	elem_type0*@
_class6
42loc:@bidirectional_rnn/fw/fw/while/dropout/truediv

Bgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/EnterEnterBgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/f_acc*
_output_shapes
:*
parallel_iterations *
T0*
is_constant(*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context

Hgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/StackPushV2StackPushV2Bgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/Enter-bidirectional_rnn/fw/fw/while/dropout/truediv^gradients/Add*
_output_shapes
: *
T0*
swap_memory( 
Ý
Ggradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/StackPopV2
StackPopV2Mgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
: 
´
Mgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/StackPopV2/EnterEnterBgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/f_acc*
_output_shapes
:*
parallel_iterations *E

frame_name75gradients/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
T0

<gradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/SumSum<gradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/MulNgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:	@*
T0*
	keep_dims( 

@gradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/ReshapeReshape<gradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Sum>gradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Shape*
_output_shapes
:	@*
Tshape0*
T0

>gradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1MulIgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/StackPopV2Sgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/tuple/control_dependency*
_output_shapes
:	@*
T0
×
Dgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/ConstConst*
_output_shapes
: *
dtype0*F
_class<
:8loc:@bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2*
valueB :
˙˙˙˙˙˙˙˙˙

Dgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/f_accStackV2Dgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/Const*
_output_shapes
:*F
_class<
:8loc:@bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2*
	elem_type0*

stack_name 
Ł
Dgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/EnterEnterDgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/f_acc*
is_constant(*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context*
parallel_iterations *
T0*
_output_shapes
:
Ą
Jgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/StackPushV2StackPushV2Dgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/Enter3bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2^gradients/Add*
swap_memory( *
_output_shapes
:	@*
T0
ę
Igradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/StackPopV2
StackPopV2Ogradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:	@
¸
Ogradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/StackPopV2/EnterEnterDgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/f_acc*
parallel_iterations *
is_constant(*
T0*E

frame_name75gradients/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:

>gradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Sum_1Sum>gradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1Pgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *
_output_shapes
: *

Tidx0
ţ
Bgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Reshape_1Reshape>gradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Sum_1@gradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Shape_1*
T0*
_output_shapes
: *
Tshape0
Ů
Igradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/tuple/group_depsNoOpA^gradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/ReshapeC^gradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Reshape_1
ę
Qgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/tuple/control_dependencyIdentity@gradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/ReshapeJ^gradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/tuple/group_deps*
_output_shapes
:	@*
T0*S
_classI
GEloc:@gradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Reshape
ç
Sgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/tuple/control_dependency_1IdentityBgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Reshape_1J^gradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/tuple/group_deps*U
_classK
IGloc:@gradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Reshape_1*
T0*
_output_shapes
: 

>gradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/MulMulNgradients/bidirectional_rnn/bw/bw/while/Select_grad/tuple/control_dependency_1Igradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul/StackPopV2*
T0*
_output_shapes
:	@
Î
Dgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul/ConstConst*
_output_shapes
: *=
_class3
1/loc:@bidirectional_rnn/bw/bw/while/dropout/Cast*
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙

Dgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul/f_accStackV2Dgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul/Const*
_output_shapes
:*
	elem_type0*

stack_name *=
_class3
1/loc:@bidirectional_rnn/bw/bw/while/dropout/Cast
Ł
Dgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul/EnterEnterDgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul/f_acc*
is_constant(*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context*
parallel_iterations *
T0*
_output_shapes
:

Jgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul/StackPushV2StackPushV2Dgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul/Enter*bidirectional_rnn/bw/bw/while/dropout/Cast^gradients/Add_1*
swap_memory( *
_output_shapes
:	@*
T0
ě
Igradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul/StackPopV2
StackPopV2Ogradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul/StackPopV2/Enter^gradients/Sub_1*
_output_shapes
:	@*
	elem_type0
¸
Ogradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul/StackPopV2/EnterEnterDgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul/f_acc*E

frame_name75gradients/bidirectional_rnn/bw/bw/while/while_context*
T0*
_output_shapes
:*
parallel_iterations *
is_constant(

@gradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul_1MulNgradients/bidirectional_rnn/bw/bw/while/Select_grad/tuple/control_dependency_1Kgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul_1/StackPopV2*
_output_shapes
:	@*
T0
Ď
Fgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul_1/ConstConst*<
_class2
0.loc:@bidirectional_rnn/bw/bw/while/dropout/mul*
dtype0*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙

Fgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul_1/f_accStackV2Fgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul_1/Const*

stack_name *<
_class2
0.loc:@bidirectional_rnn/bw/bw/while/dropout/mul*
_output_shapes
:*
	elem_type0
§
Fgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul_1/EnterEnterFgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul_1/f_acc*
T0*
_output_shapes
:*
parallel_iterations *;

frame_name-+bidirectional_rnn/bw/bw/while/while_context*
is_constant(

Lgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul_1/StackPushV2StackPushV2Fgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul_1/Enter)bidirectional_rnn/bw/bw/while/dropout/mul^gradients/Add_1*
_output_shapes
:	@*
T0*
swap_memory( 
đ
Kgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul_1/StackPopV2
StackPopV2Qgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul_1/StackPopV2/Enter^gradients/Sub_1*
	elem_type0*
_output_shapes
:	@
ź
Qgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul_1/StackPopV2/EnterEnterFgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul_1/f_acc*
T0*
_output_shapes
:*
is_constant(*
parallel_iterations *E

frame_name75gradients/bidirectional_rnn/bw/bw/while/while_context
×
Kgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/tuple/group_depsNoOp?^gradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/MulA^gradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul_1
ę
Sgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/tuple/control_dependencyIdentity>gradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/MulL^gradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/tuple/group_deps*
_output_shapes
:	@*
T0*Q
_classG
ECloc:@gradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul
đ
Ugradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/tuple/control_dependency_1Identity@gradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul_1L^gradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/tuple/group_deps*S
_classI
GEloc:@gradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul_1*
_output_shapes
:	@*
T0
ä
Egradients/bidirectional_rnn/bw/bw/while/Switch_2_grad_1/NextIterationNextIterationkgradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1*
T0*
_output_shapes
: 
Á
gradients/AddNAddNPgradients/bidirectional_rnn/fw/fw/while/Select_2_grad/tuple/control_dependency_1Qgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/tuple/control_dependency*
T0*
_output_shapes
:	@*Q
_classG
ECloc:@gradients/bidirectional_rnn/fw/fw/while/Select_2_grad/Select_1*
N
Ú
Fgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/MulMulgradients/AddNQgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2*
_output_shapes
:	@*
T0
ă
Lgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul/ConstConst*
valueB :
˙˙˙˙˙˙˙˙˙*J
_class@
><loc:@bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid_2*
dtype0*
_output_shapes
: 
°
Lgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul/f_accStackV2Lgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul/Const*

stack_name *J
_class@
><loc:@bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid_2*
_output_shapes
:*
	elem_type0
ł
Lgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul/EnterEnterLgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul/f_acc*
is_constant(*
parallel_iterations *;

frame_name-+bidirectional_rnn/fw/fw/while/while_context*
T0*
_output_shapes
:
ľ
Rgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul/StackPushV2StackPushV2Lgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul/Enter7bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid_2^gradients/Add*
swap_memory( *
_output_shapes
:	@*
T0
ú
Qgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2
StackPopV2Wgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2/Enter^gradients/Sub*
_output_shapes
:	@*
	elem_type0
Č
Wgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2/EnterEnterLgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul/f_acc*
is_constant(*
T0*
parallel_iterations *E

frame_name75gradients/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
Ţ
Hgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul_1Mulgradients/AddNSgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2*
_output_shapes
:	@*
T0
â
Ngradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul_1/ConstConst*
valueB :
˙˙˙˙˙˙˙˙˙*
_output_shapes
: *G
_class=
;9loc:@bidirectional_rnn/fw/fw/while/basic_lstm_cell/Tanh_1*
dtype0
ą
Ngradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul_1/f_accStackV2Ngradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul_1/Const*

stack_name *
	elem_type0*
_output_shapes
:*G
_class=
;9loc:@bidirectional_rnn/fw/fw/while/basic_lstm_cell/Tanh_1
ˇ
Ngradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul_1/EnterEnterNgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul_1/f_acc*
_output_shapes
:*
is_constant(*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context*
T0*
parallel_iterations 
ś
Tgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPushV2StackPushV2Ngradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul_1/Enter4bidirectional_rnn/fw/fw/while/basic_lstm_cell/Tanh_1^gradients/Add*
swap_memory( *
T0*
_output_shapes
:	@
ţ
Sgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2
StackPopV2Ygradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:	@
Ě
Ygradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2/EnterEnterNgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul_1/f_acc*
is_constant(*E

frame_name75gradients/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:*
T0*
parallel_iterations 
ď
Sgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/tuple/group_depsNoOpG^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/MulI^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul_1

[gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/tuple/control_dependencyIdentityFgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/MulT^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/tuple/group_deps*
_output_shapes
:	@*
T0*Y
_classO
MKloc:@gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul

]gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/tuple/control_dependency_1IdentityHgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul_1T^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/tuple/group_deps*
_output_shapes
:	@*
T0*[
_classQ
OMloc:@gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul_1
Ą
>gradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/ShapeConst^gradients/Sub_1*
valueB"   @   *
dtype0*
_output_shapes
:

@gradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Shape_1Const^gradients/Sub_1*
valueB *
dtype0*
_output_shapes
: 
Ś
Ngradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs>gradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Shape@gradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

<gradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/MulMulSgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/tuple/control_dependencyGgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/StackPopV2*
_output_shapes
:	@*
T0
Ď
Bgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/ConstConst*
_output_shapes
: *@
_class6
42loc:@bidirectional_rnn/bw/bw/while/dropout/truediv*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0

Bgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/f_accStackV2Bgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/Const*

stack_name *
	elem_type0*
_output_shapes
:*@
_class6
42loc:@bidirectional_rnn/bw/bw/while/dropout/truediv

Bgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/EnterEnterBgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/f_acc*
T0*
_output_shapes
:*
is_constant(*
parallel_iterations *;

frame_name-+bidirectional_rnn/bw/bw/while/while_context

Hgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/StackPushV2StackPushV2Bgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/Enter-bidirectional_rnn/bw/bw/while/dropout/truediv^gradients/Add_1*
T0*
_output_shapes
: *
swap_memory( 
ß
Ggradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/StackPopV2
StackPopV2Mgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/StackPopV2/Enter^gradients/Sub_1*
_output_shapes
: *
	elem_type0
´
Mgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/StackPopV2/EnterEnterBgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*
T0*E

frame_name75gradients/bidirectional_rnn/bw/bw/while/while_context

<gradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/SumSum<gradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/MulNgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:	@

@gradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/ReshapeReshape<gradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Sum>gradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Shape*
Tshape0*
_output_shapes
:	@*
T0

>gradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1MulIgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/StackPopV2Sgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/tuple/control_dependency*
_output_shapes
:	@*
T0
×
Dgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/ConstConst*F
_class<
:8loc:@bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2*
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙*
_output_shapes
: 

Dgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/f_accStackV2Dgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/Const*F
_class<
:8loc:@bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2*
	elem_type0*
_output_shapes
:*

stack_name 
Ł
Dgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/EnterEnterDgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/f_acc*
is_constant(*
T0*
_output_shapes
:*
parallel_iterations *;

frame_name-+bidirectional_rnn/bw/bw/while/while_context
Ł
Jgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/StackPushV2StackPushV2Dgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/Enter3bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2^gradients/Add_1*
T0*
_output_shapes
:	@*
swap_memory( 
ě
Igradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/StackPopV2
StackPopV2Ogradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/StackPopV2/Enter^gradients/Sub_1*
	elem_type0*
_output_shapes
:	@
¸
Ogradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/StackPopV2/EnterEnterDgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/f_acc*
is_constant(*
T0*E

frame_name75gradients/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:*
parallel_iterations 

>gradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Sum_1Sum>gradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1Pgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs:1*
_output_shapes
: *
	keep_dims( *
T0*

Tidx0
ţ
Bgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Reshape_1Reshape>gradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Sum_1@gradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Ů
Igradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/tuple/group_depsNoOpA^gradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/ReshapeC^gradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Reshape_1
ę
Qgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/tuple/control_dependencyIdentity@gradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/ReshapeJ^gradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Reshape*
_output_shapes
:	@
ç
Sgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/tuple/control_dependency_1IdentityBgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Reshape_1J^gradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/tuple/group_deps*
_output_shapes
: *U
_classK
IGloc:@gradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Reshape_1*
T0
´
Lgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Tanh_1_grad/TanhGradTanhGradSgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2[gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/tuple/control_dependency*
_output_shapes
:	@*
T0
˝
Rgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGradQgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2]gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/tuple/control_dependency_1*
_output_shapes
:	@*
T0
Ă
gradients/AddN_1AddNPgradients/bidirectional_rnn/bw/bw/while/Select_2_grad/tuple/control_dependency_1Qgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/tuple/control_dependency*
N*
T0*Q
_classG
ECloc:@gradients/bidirectional_rnn/bw/bw/while/Select_2_grad/Select_1*
_output_shapes
:	@
Ü
Fgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/MulMulgradients/AddN_1Qgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2*
_output_shapes
:	@*
T0
ă
Lgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul/ConstConst*
valueB :
˙˙˙˙˙˙˙˙˙*J
_class@
><loc:@bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid_2*
dtype0*
_output_shapes
: 
°
Lgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul/f_accStackV2Lgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul/Const*
	elem_type0*
_output_shapes
:*J
_class@
><loc:@bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid_2*

stack_name 
ł
Lgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul/EnterEnterLgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul/f_acc*
T0*
parallel_iterations *
_output_shapes
:*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context*
is_constant(
ˇ
Rgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul/StackPushV2StackPushV2Lgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul/Enter7bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid_2^gradients/Add_1*
swap_memory( *
_output_shapes
:	@*
T0
ü
Qgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2
StackPopV2Wgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2/Enter^gradients/Sub_1*
	elem_type0*
_output_shapes
:	@
Č
Wgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2/EnterEnterLgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul/f_acc*
T0*E

frame_name75gradients/bidirectional_rnn/bw/bw/while/while_context*
parallel_iterations *
_output_shapes
:*
is_constant(
ŕ
Hgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul_1Mulgradients/AddN_1Sgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2*
_output_shapes
:	@*
T0
â
Ngradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul_1/ConstConst*G
_class=
;9loc:@bidirectional_rnn/bw/bw/while/basic_lstm_cell/Tanh_1*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
ą
Ngradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul_1/f_accStackV2Ngradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul_1/Const*
	elem_type0*G
_class=
;9loc:@bidirectional_rnn/bw/bw/while/basic_lstm_cell/Tanh_1*

stack_name *
_output_shapes
:
ˇ
Ngradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul_1/EnterEnterNgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul_1/f_acc*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context*
T0*
_output_shapes
:*
parallel_iterations *
is_constant(
¸
Tgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPushV2StackPushV2Ngradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul_1/Enter4bidirectional_rnn/bw/bw/while/basic_lstm_cell/Tanh_1^gradients/Add_1*
T0*
swap_memory( *
_output_shapes
:	@

Sgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2
StackPopV2Ygradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2/Enter^gradients/Sub_1*
	elem_type0*
_output_shapes
:	@
Ě
Ygradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2/EnterEnterNgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul_1/f_acc*
is_constant(*
_output_shapes
:*E

frame_name75gradients/bidirectional_rnn/bw/bw/while/while_context*
T0*
parallel_iterations 
ď
Sgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/tuple/group_depsNoOpG^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/MulI^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul_1

[gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/tuple/control_dependencyIdentityFgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/MulT^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/tuple/group_deps*
_output_shapes
:	@*
T0*Y
_classO
MKloc:@gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul

]gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/tuple/control_dependency_1IdentityHgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul_1T^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/tuple/group_deps*[
_classQ
OMloc:@gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul_1*
_output_shapes
:	@*
T0
ž
gradients/AddN_2AddNPgradients/bidirectional_rnn/fw/fw/while/Select_1_grad/tuple/control_dependency_1Lgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Tanh_1_grad/TanhGrad*
N*Q
_classG
ECloc:@gradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select_1*
T0*
_output_shapes
:	@
n
Sgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_1_grad/tuple/group_depsNoOp^gradients/AddN_2
Ě
[gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_1_grad/tuple/control_dependencyIdentitygradients/AddN_2T^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_1_grad/tuple/group_deps*
_output_shapes
:	@*
T0*Q
_classG
ECloc:@gradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select_1
Î
]gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_1_grad/tuple/control_dependency_1Identitygradients/AddN_2T^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_1_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select_1*
_output_shapes
:	@
´
Lgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Tanh_1_grad/TanhGradTanhGradSgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2[gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/tuple/control_dependency*
T0*
_output_shapes
:	@
˝
Rgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGradQgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2]gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/tuple/control_dependency_1*
T0*
_output_shapes
:	@
Ł
Dgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/MulMul[gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_1_grad/tuple/control_dependencyOgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2*
T0*
_output_shapes
:	@
ß
Jgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul/ConstConst*
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙*H
_class>
<:loc:@bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid*
_output_shapes
: 
Ş
Jgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul/f_accStackV2Jgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul/Const*
_output_shapes
:*
	elem_type0*H
_class>
<:loc:@bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid*

stack_name 
Ż
Jgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul/EnterEnterJgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul/f_acc*
is_constant(*
_output_shapes
:*
T0*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context*
parallel_iterations 
Ż
Pgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul/StackPushV2StackPushV2Jgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul/Enter5bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid^gradients/Add*
T0*
_output_shapes
:	@*
swap_memory( 
ö
Ogradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2
StackPopV2Ugradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:	@
Ä
Ugradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2/EnterEnterJgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul/f_acc*E

frame_name75gradients/bidirectional_rnn/fw/fw/while/while_context*
T0*
is_constant(*
_output_shapes
:*
parallel_iterations 
§
Fgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul_1Mul[gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_1_grad/tuple/control_dependencyQgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2*
_output_shapes
:	@*
T0
Ô
Lgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul_1/ConstConst*
dtype0*
_output_shapes
: *;
_class1
/-loc:@bidirectional_rnn/fw/fw/while/Identity_3*
valueB :
˙˙˙˙˙˙˙˙˙
Ą
Lgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul_1/f_accStackV2Lgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul_1/Const*
	elem_type0*
_output_shapes
:*

stack_name *;
_class1
/-loc:@bidirectional_rnn/fw/fw/while/Identity_3
ł
Lgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul_1/EnterEnterLgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul_1/f_acc*
T0*
parallel_iterations *
_output_shapes
:*
is_constant(*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context
Ś
Rgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul_1/StackPushV2StackPushV2Lgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul_1/Enter(bidirectional_rnn/fw/fw/while/Identity_3^gradients/Add*
_output_shapes
:	@*
swap_memory( *
T0
ú
Qgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2
StackPopV2Wgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:	@
Č
Wgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2/EnterEnterLgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul_1/f_acc*
is_constant(*
T0*
parallel_iterations *
_output_shapes
:*E

frame_name75gradients/bidirectional_rnn/fw/fw/while/while_context
é
Qgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/tuple/group_depsNoOpE^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/MulG^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul_1

Ygradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/tuple/control_dependencyIdentityDgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/MulR^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul*
_output_shapes
:	@

[gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/tuple/control_dependency_1IdentityFgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul_1R^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul_1*
_output_shapes
:	@
Š
Fgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/MulMul]gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_1_grad/tuple/control_dependency_1Qgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2*
_output_shapes
:	@*
T0
Ţ
Lgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul/ConstConst*
dtype0*
_output_shapes
: *E
_class;
97loc:@bidirectional_rnn/fw/fw/while/basic_lstm_cell/Tanh*
valueB :
˙˙˙˙˙˙˙˙˙
Ť
Lgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul/f_accStackV2Lgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul/Const*E
_class;
97loc:@bidirectional_rnn/fw/fw/while/basic_lstm_cell/Tanh*
_output_shapes
:*

stack_name *
	elem_type0
ł
Lgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul/EnterEnterLgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul/f_acc*
_output_shapes
:*
parallel_iterations *
is_constant(*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context*
T0
°
Rgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul/StackPushV2StackPushV2Lgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul/Enter2bidirectional_rnn/fw/fw/while/basic_lstm_cell/Tanh^gradients/Add*
_output_shapes
:	@*
T0*
swap_memory( 
ú
Qgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2
StackPopV2Wgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2/Enter^gradients/Sub*
_output_shapes
:	@*
	elem_type0
Č
Wgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2/EnterEnterLgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul/f_acc*
T0*E

frame_name75gradients/bidirectional_rnn/fw/fw/while/while_context*
parallel_iterations *
is_constant(*
_output_shapes
:
­
Hgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul_1Mul]gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_1_grad/tuple/control_dependency_1Sgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2*
T0*
_output_shapes
:	@
ĺ
Ngradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul_1/ConstConst*
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙*J
_class@
><loc:@bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid_1*
_output_shapes
: 
´
Ngradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul_1/f_accStackV2Ngradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul_1/Const*
_output_shapes
:*J
_class@
><loc:@bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid_1*

stack_name *
	elem_type0
ˇ
Ngradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul_1/EnterEnterNgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul_1/f_acc*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context*
parallel_iterations *
is_constant(*
_output_shapes
:*
T0
š
Tgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPushV2StackPushV2Ngradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul_1/Enter7bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid_1^gradients/Add*
_output_shapes
:	@*
T0*
swap_memory( 
ţ
Sgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2
StackPopV2Ygradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:	@
Ě
Ygradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2/EnterEnterNgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul_1/f_acc*E

frame_name75gradients/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:*
T0*
parallel_iterations *
is_constant(
ď
Sgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/tuple/group_depsNoOpG^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/MulI^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul_1

[gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/tuple/control_dependencyIdentityFgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/MulT^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/tuple/group_deps*
_output_shapes
:	@*Y
_classO
MKloc:@gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul*
T0

]gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/tuple/control_dependency_1IdentityHgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul_1T^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/tuple/group_deps*
_output_shapes
:	@*[
_classQ
OMloc:@gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul_1*
T0
ž
gradients/AddN_3AddNPgradients/bidirectional_rnn/bw/bw/while/Select_1_grad/tuple/control_dependency_1Lgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Tanh_1_grad/TanhGrad*Q
_classG
ECloc:@gradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select_1*
_output_shapes
:	@*
N*
T0
n
Sgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_1_grad/tuple/group_depsNoOp^gradients/AddN_3
Ě
[gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_1_grad/tuple/control_dependencyIdentitygradients/AddN_3T^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_1_grad/tuple/group_deps*
_output_shapes
:	@*
T0*Q
_classG
ECloc:@gradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select_1
Î
]gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_1_grad/tuple/control_dependency_1Identitygradients/AddN_3T^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_1_grad/tuple/group_deps*Q
_classG
ECloc:@gradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select_1*
_output_shapes
:	@*
T0
Ç
gradients/AddN_4AddNNgradients/bidirectional_rnn/fw/fw/while/Select_1_grad/tuple/control_dependencyYgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/tuple/control_dependency*
N*O
_classE
CAloc:@gradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select*
T0*
_output_shapes
:	@
ˇ
Pgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGradOgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2[gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/tuple/control_dependency_1*
_output_shapes
:	@*
T0
˝
Rgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGradSgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2[gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/tuple/control_dependency*
T0*
_output_shapes
:	@
˛
Jgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Tanh_grad/TanhGradTanhGradQgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2]gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/tuple/control_dependency_1*
T0*
_output_shapes
:	@
Ł
Dgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/MulMul[gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_1_grad/tuple/control_dependencyOgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2*
T0*
_output_shapes
:	@
ß
Jgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul/ConstConst*
dtype0*H
_class>
<:loc:@bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid*
valueB :
˙˙˙˙˙˙˙˙˙*
_output_shapes
: 
Ş
Jgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul/f_accStackV2Jgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul/Const*
_output_shapes
:*H
_class>
<:loc:@bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid*
	elem_type0*

stack_name 
Ż
Jgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul/EnterEnterJgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul/f_acc*
parallel_iterations *;

frame_name-+bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:*
is_constant(*
T0
ą
Pgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul/StackPushV2StackPushV2Jgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul/Enter5bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid^gradients/Add_1*
T0*
swap_memory( *
_output_shapes
:	@
ř
Ogradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2
StackPopV2Ugradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2/Enter^gradients/Sub_1*
	elem_type0*
_output_shapes
:	@
Ä
Ugradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2/EnterEnterJgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul/f_acc*
parallel_iterations *
T0*
_output_shapes
:*
is_constant(*E

frame_name75gradients/bidirectional_rnn/bw/bw/while/while_context
§
Fgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul_1Mul[gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_1_grad/tuple/control_dependencyQgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2*
_output_shapes
:	@*
T0
Ô
Lgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul_1/ConstConst*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*;
_class1
/-loc:@bidirectional_rnn/bw/bw/while/Identity_3
Ą
Lgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul_1/f_accStackV2Lgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul_1/Const*

stack_name *
_output_shapes
:*
	elem_type0*;
_class1
/-loc:@bidirectional_rnn/bw/bw/while/Identity_3
ł
Lgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul_1/EnterEnterLgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul_1/f_acc*
T0*
_output_shapes
:*
parallel_iterations *
is_constant(*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context
¨
Rgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul_1/StackPushV2StackPushV2Lgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul_1/Enter(bidirectional_rnn/bw/bw/while/Identity_3^gradients/Add_1*
T0*
_output_shapes
:	@*
swap_memory( 
ü
Qgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2
StackPopV2Wgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2/Enter^gradients/Sub_1*
_output_shapes
:	@*
	elem_type0
Č
Wgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2/EnterEnterLgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul_1/f_acc*
is_constant(*
T0*
parallel_iterations *
_output_shapes
:*E

frame_name75gradients/bidirectional_rnn/bw/bw/while/while_context
é
Qgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/tuple/group_depsNoOpE^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/MulG^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul_1

Ygradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/tuple/control_dependencyIdentityDgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/MulR^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/tuple/group_deps*W
_classM
KIloc:@gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul*
_output_shapes
:	@*
T0

[gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/tuple/control_dependency_1IdentityFgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul_1R^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/tuple/group_deps*Y
_classO
MKloc:@gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul_1*
T0*
_output_shapes
:	@
Š
Fgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/MulMul]gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_1_grad/tuple/control_dependency_1Qgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2*
_output_shapes
:	@*
T0
Ţ
Lgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul/ConstConst*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*E
_class;
97loc:@bidirectional_rnn/bw/bw/while/basic_lstm_cell/Tanh
Ť
Lgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul/f_accStackV2Lgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul/Const*
_output_shapes
:*

stack_name *E
_class;
97loc:@bidirectional_rnn/bw/bw/while/basic_lstm_cell/Tanh*
	elem_type0
ł
Lgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul/EnterEnterLgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul/f_acc*
_output_shapes
:*
parallel_iterations *
is_constant(*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context*
T0
˛
Rgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul/StackPushV2StackPushV2Lgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul/Enter2bidirectional_rnn/bw/bw/while/basic_lstm_cell/Tanh^gradients/Add_1*
_output_shapes
:	@*
swap_memory( *
T0
ü
Qgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2
StackPopV2Wgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2/Enter^gradients/Sub_1*
_output_shapes
:	@*
	elem_type0
Č
Wgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2/EnterEnterLgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul/f_acc*
T0*
parallel_iterations *E

frame_name75gradients/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:*
is_constant(
­
Hgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul_1Mul]gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_1_grad/tuple/control_dependency_1Sgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2*
_output_shapes
:	@*
T0
ĺ
Ngradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul_1/ConstConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙*J
_class@
><loc:@bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid_1
´
Ngradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul_1/f_accStackV2Ngradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul_1/Const*
_output_shapes
:*
	elem_type0*J
_class@
><loc:@bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid_1*

stack_name 
ˇ
Ngradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul_1/EnterEnterNgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul_1/f_acc*
parallel_iterations *;

frame_name-+bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:*
is_constant(*
T0
ť
Tgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPushV2StackPushV2Ngradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul_1/Enter7bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid_1^gradients/Add_1*
_output_shapes
:	@*
T0*
swap_memory( 

Sgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2
StackPopV2Ygradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2/Enter^gradients/Sub_1*
	elem_type0*
_output_shapes
:	@
Ě
Ygradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2/EnterEnterNgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul_1/f_acc*
T0*E

frame_name75gradients/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:*
parallel_iterations 
ď
Sgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/tuple/group_depsNoOpG^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/MulI^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul_1

[gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/tuple/control_dependencyIdentityFgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/MulT^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul*
_output_shapes
:	@

]gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/tuple/control_dependency_1IdentityHgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul_1T^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/tuple/group_deps*[
_classQ
OMloc:@gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul_1*
T0*
_output_shapes
:	@

Egradients/bidirectional_rnn/fw/fw/while/Switch_3_grad_1/NextIterationNextIterationgradients/AddN_4*
_output_shapes
:	@*
T0
§
Fgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_grad/ShapeConst^gradients/Sub*
valueB"   @   *
_output_shapes
:*
dtype0

Hgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_grad/Shape_1Const^gradients/Sub*
valueB *
dtype0*
_output_shapes
: 
ž
Vgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_grad/BroadcastGradientArgsBroadcastGradientArgsFgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_grad/ShapeHgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
ź
Dgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_grad/SumSumPgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid_grad/SigmoidGradVgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:	@

Hgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_grad/ReshapeReshapeDgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_grad/SumFgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_grad/Shape*
_output_shapes
:	@*
T0*
Tshape0
ˇ
Fgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_grad/Sum_1SumPgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid_grad/SigmoidGradXgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs:1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0

Jgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_grad/Reshape_1ReshapeFgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_grad/Sum_1Hgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
ń
Qgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_grad/tuple/group_depsNoOpI^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_grad/ReshapeK^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_grad/Reshape_1

Ygradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_grad/tuple/control_dependencyIdentityHgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_grad/ReshapeR^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_grad/tuple/group_deps*
_output_shapes
:	@*
T0*[
_classQ
OMloc:@gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_grad/Reshape

[gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_grad/tuple/control_dependency_1IdentityJgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_grad/Reshape_1R^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_grad/Reshape_1*
_output_shapes
: 
Ç
gradients/AddN_5AddNNgradients/bidirectional_rnn/bw/bw/while/Select_1_grad/tuple/control_dependencyYgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/tuple/control_dependency*
T0*O
_classE
CAloc:@gradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select*
_output_shapes
:	@*
N
ˇ
Pgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGradOgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2[gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/tuple/control_dependency_1*
_output_shapes
:	@*
T0
˝
Rgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGradSgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2[gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/tuple/control_dependency*
T0*
_output_shapes
:	@
˛
Jgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Tanh_grad/TanhGradTanhGradQgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2]gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/tuple/control_dependency_1*
_output_shapes
:	@*
T0
ľ
Igradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/split_grad/concatConcatV2Rgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid_1_grad/SigmoidGradJgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Tanh_grad/TanhGradYgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_grad/tuple/control_dependencyRgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid_2_grad/SigmoidGradOgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/split_grad/concat/Const* 
_output_shapes
:
*
N*
T0*

Tidx0
Ą
Ogradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/split_grad/concat/ConstConst^gradients/Sub*
dtype0*
_output_shapes
: *
value	B :

Egradients/bidirectional_rnn/bw/bw/while/Switch_3_grad_1/NextIterationNextIterationgradients/AddN_5*
_output_shapes
:	@*
T0
Š
Fgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_grad/ShapeConst^gradients/Sub_1*
dtype0*
_output_shapes
:*
valueB"   @   

Hgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_grad/Shape_1Const^gradients/Sub_1*
dtype0*
valueB *
_output_shapes
: 
ž
Vgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_grad/BroadcastGradientArgsBroadcastGradientArgsFgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_grad/ShapeHgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
ź
Dgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_grad/SumSumPgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid_grad/SigmoidGradVgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
_output_shapes
:	@*
T0

Hgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_grad/ReshapeReshapeDgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_grad/SumFgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_grad/Shape*
_output_shapes
:	@*
T0*
Tshape0
ˇ
Fgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_grad/Sum_1SumPgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid_grad/SigmoidGradXgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
: 

Jgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_grad/Reshape_1ReshapeFgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_grad/Sum_1Hgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
ń
Qgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_grad/tuple/group_depsNoOpI^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_grad/ReshapeK^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_grad/Reshape_1

Ygradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_grad/tuple/control_dependencyIdentityHgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_grad/ReshapeR^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_grad/tuple/group_deps*[
_classQ
OMloc:@gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_grad/Reshape*
_output_shapes
:	@*
T0

[gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_grad/tuple/control_dependency_1IdentityJgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_grad/Reshape_1R^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_grad/tuple/group_deps*]
_classS
QOloc:@gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_grad/Reshape_1*
T0*
_output_shapes
: 
ç
Pgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGradIgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/split_grad/concat*
_output_shapes	
:*
T0*
data_formatNHWC
ü
Ugradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd_grad/tuple/group_depsNoOpQ^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd_grad/BiasAddGradJ^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/split_grad/concat

]gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependencyIdentityIgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/split_grad/concatV^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd_grad/tuple/group_deps* 
_output_shapes
:
*\
_classR
PNloc:@gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/split_grad/concat*
T0
 
_gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependency_1IdentityPgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd_grad/BiasAddGradV^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd_grad/tuple/group_deps*c
_classY
WUloc:@gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
ľ
Igradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/split_grad/concatConcatV2Rgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid_1_grad/SigmoidGradJgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Tanh_grad/TanhGradYgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_grad/tuple/control_dependencyRgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid_2_grad/SigmoidGradOgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/split_grad/concat/Const*

Tidx0*
T0* 
_output_shapes
:
*
N
Ł
Ogradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/split_grad/concat/ConstConst^gradients/Sub_1*
_output_shapes
: *
value	B :*
dtype0
Ö
Jgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/MatMulMatMul]gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependencyPgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/MatMul/Enter*
transpose_a( *
transpose_b(* 
_output_shapes
:
Ŕ*
T0
Ť
Pgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/MatMul/EnterEnter0bidirectional_rnn/fw/basic_lstm_cell/kernel/read* 
_output_shapes
:
Ŕ*
T0*
parallel_iterations *
is_constant(*E

frame_name75gradients/bidirectional_rnn/fw/fw/while/while_context
ß
Lgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/MatMul_1MatMulWgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2]gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
Ŕ*
transpose_a(*
transpose_b( *
T0
ć
Rgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/MatMul_1/ConstConst*
_output_shapes
: *G
_class=
;9loc:@bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
š
Rgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/MatMul_1/f_accStackV2Rgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/MatMul_1/Const*
	elem_type0*G
_class=
;9loc:@bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat*
_output_shapes
:*

stack_name 
ż
Rgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/MatMul_1/EnterEnterRgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
_output_shapes
:*
is_constant(*
parallel_iterations *;

frame_name-+bidirectional_rnn/fw/fw/while/while_context
ż
Xgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2StackPushV2Rgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/MatMul_1/Enter4bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat^gradients/Add*
T0*
swap_memory( * 
_output_shapes
:
Ŕ

Wgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2
StackPopV2]gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2/Enter^gradients/Sub* 
_output_shapes
:
Ŕ*
	elem_type0
Ô
]gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2/EnterEnterRgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc*
is_constant(*
parallel_iterations *
_output_shapes
:*E

frame_name75gradients/bidirectional_rnn/fw/fw/while/while_context*
T0
ř
Tgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/tuple/group_depsNoOpK^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/MatMulM^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/MatMul_1

\gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/tuple/control_dependencyIdentityJgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/MatMulU^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/tuple/group_deps* 
_output_shapes
:
Ŕ*]
_classS
QOloc:@gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/MatMul*
T0

^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/tuple/control_dependency_1IdentityLgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/MatMul_1U^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/tuple/group_deps* 
_output_shapes
:
Ŕ*_
_classU
SQloc:@gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/MatMul_1*
T0

Pgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd/Enter_grad/b_accConst*
_output_shapes	
:*
valueB*    *
dtype0
Č
Rgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1EnterPgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc*E

frame_name75gradients/bidirectional_rnn/fw/fw/while/while_context*
is_constant( *
parallel_iterations *
_output_shapes	
:*
T0
ş
Rgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2MergeRgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1Xgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd/Enter_grad/NextIteration*
N*
_output_shapes
	:: *
T0
ń
Qgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd/Enter_grad/SwitchSwitchRgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2gradients/b_count_2*"
_output_shapes
::*
T0
ą
Ngradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd/Enter_grad/AddAddSgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd/Enter_grad/Switch:1_gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes	
:*
T0
ß
Xgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd/Enter_grad/NextIterationNextIterationNgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd/Enter_grad/Add*
T0*
_output_shapes	
:
Ó
Rgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3ExitQgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd/Enter_grad/Switch*
T0*
_output_shapes	
:
ç
Pgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGradIgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/split_grad/concat*
_output_shapes	
:*
T0*
data_formatNHWC
ü
Ugradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd_grad/tuple/group_depsNoOpQ^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd_grad/BiasAddGradJ^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/split_grad/concat

]gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependencyIdentityIgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/split_grad/concatV^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd_grad/tuple/group_deps* 
_output_shapes
:
*
T0*\
_classR
PNloc:@gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/split_grad/concat
 
_gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependency_1IdentityPgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd_grad/BiasAddGradV^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd_grad/tuple/group_deps*c
_classY
WUloc:@gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0

Igradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/ConstConst^gradients/Sub*
_output_shapes
: *
dtype0*
value	B :

Hgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/RankConst^gradients/Sub*
value	B :*
_output_shapes
: *
dtype0

Ggradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/modFloorModIgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/ConstHgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/Rank*
T0*
_output_shapes
: 
Ş
Igradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
dtype0*
valueB"      
Ź
Kgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/Shape_1Const^gradients/Sub*
dtype0*
_output_shapes
:*
valueB"   @   
ě
Pgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/ConcatOffsetConcatOffsetGgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/modIgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/ShapeKgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/Shape_1* 
_output_shapes
::*
N

Igradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/SliceSlice\gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/tuple/control_dependencyPgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/ConcatOffsetIgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/Shape*
Index0* 
_output_shapes
:
*
T0

Kgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/Slice_1Slice\gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/tuple/control_dependencyRgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/ConcatOffset:1Kgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/Shape_1*
T0*
_output_shapes
:	@*
Index0
ö
Tgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/tuple/group_depsNoOpJ^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/SliceL^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/Slice_1

\gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/tuple/control_dependencyIdentityIgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/SliceU^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/tuple/group_deps* 
_output_shapes
:
*
T0*\
_classR
PNloc:@gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/Slice

^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/tuple/control_dependency_1IdentityKgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/Slice_1U^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/Slice_1*
_output_shapes
:	@
¨
Ogradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul/Enter_grad/b_accConst* 
_output_shapes
:
Ŕ*
valueB
Ŕ*    *
dtype0
Ë
Qgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_1EnterOgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul/Enter_grad/b_acc* 
_output_shapes
:
Ŕ*
parallel_iterations *
T0*E

frame_name75gradients/bidirectional_rnn/fw/fw/while/while_context*
is_constant( 
ź
Qgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_2MergeQgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_1Wgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul/Enter_grad/NextIteration*"
_output_shapes
:
Ŕ: *
T0*
N
ů
Pgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul/Enter_grad/SwitchSwitchQgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_2gradients/b_count_2*,
_output_shapes
:
Ŕ:
Ŕ*
T0
ł
Mgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul/Enter_grad/AddAddRgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul/Enter_grad/Switch:1^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:
Ŕ
â
Wgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul/Enter_grad/NextIterationNextIterationMgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul/Enter_grad/Add*
T0* 
_output_shapes
:
Ŕ
Ö
Qgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_3ExitPgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul/Enter_grad/Switch*
T0* 
_output_shapes
:
Ŕ
Ö
Jgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/MatMulMatMul]gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependencyPgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/MatMul/Enter* 
_output_shapes
:
Ŕ*
T0*
transpose_b(*
transpose_a( 
Ť
Pgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/MatMul/EnterEnter0bidirectional_rnn/bw/basic_lstm_cell/kernel/read* 
_output_shapes
:
Ŕ*
parallel_iterations *E

frame_name75gradients/bidirectional_rnn/bw/bw/while/while_context*
T0*
is_constant(
ß
Lgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/MatMul_1MatMulWgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2]gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
Ŕ*
transpose_a(*
transpose_b( 
ć
Rgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/MatMul_1/ConstConst*G
_class=
;9loc:@bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
š
Rgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/MatMul_1/f_accStackV2Rgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/MatMul_1/Const*G
_class=
;9loc:@bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat*

stack_name *
_output_shapes
:*
	elem_type0
ż
Rgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/MatMul_1/EnterEnterRgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc*
parallel_iterations *
_output_shapes
:*
T0*
is_constant(*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context
Á
Xgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2StackPushV2Rgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/MatMul_1/Enter4bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat^gradients/Add_1* 
_output_shapes
:
Ŕ*
swap_memory( *
T0

Wgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2
StackPopV2]gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2/Enter^gradients/Sub_1*
	elem_type0* 
_output_shapes
:
Ŕ
Ô
]gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2/EnterEnterRgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc*
_output_shapes
:*
parallel_iterations *
is_constant(*E

frame_name75gradients/bidirectional_rnn/bw/bw/while/while_context*
T0
ř
Tgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/tuple/group_depsNoOpK^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/MatMulM^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/MatMul_1

\gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/tuple/control_dependencyIdentityJgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/MatMulU^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/tuple/group_deps* 
_output_shapes
:
Ŕ*
T0*]
_classS
QOloc:@gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/MatMul

^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/tuple/control_dependency_1IdentityLgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/MatMul_1U^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/tuple/group_deps* 
_output_shapes
:
Ŕ*_
_classU
SQloc:@gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/MatMul_1*
T0

Pgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd/Enter_grad/b_accConst*
dtype0*
valueB*    *
_output_shapes	
:
Č
Rgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1EnterPgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc*
parallel_iterations *
_output_shapes	
:*E

frame_name75gradients/bidirectional_rnn/bw/bw/while/while_context*
T0*
is_constant( 
ş
Rgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2MergeRgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1Xgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd/Enter_grad/NextIteration*
N*
_output_shapes
	:: *
T0
ń
Qgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd/Enter_grad/SwitchSwitchRgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2gradients/b_count_6*"
_output_shapes
::*
T0
ą
Ngradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd/Enter_grad/AddAddSgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd/Enter_grad/Switch:1_gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:
ß
Xgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd/Enter_grad/NextIterationNextIterationNgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd/Enter_grad/Add*
_output_shapes	
:*
T0
Ó
Rgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3ExitQgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd/Enter_grad/Switch*
_output_shapes	
:*
T0
Ö
`gradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3fgradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enterhgradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1^gradients/Sub*H
_class>
<:loc:@bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter*
_output_shapes

:: *
source	gradients
ú
fgradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnter%bidirectional_rnn/fw/fw/TensorArray_1*
is_constant(*E

frame_name75gradients/bidirectional_rnn/fw/fw/while/while_context*H
_class>
<:loc:@bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter*
T0*
_output_shapes
:*
parallel_iterations 
Ľ
hgradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1EnterRbidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations *H
_class>
<:loc:@bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter*
_output_shapes
: *E

frame_name75gradients/bidirectional_rnn/fw/fw/while/while_context
 
\gradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flowIdentityhgradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1a^gradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3*H
_class>
<:loc:@bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter*
T0*
_output_shapes
: 
Ś
bgradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3`gradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3mgradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2\gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/tuple/control_dependency\gradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flow*
T0*
_output_shapes
: 
Ě
gradients/AddN_6AddNNgradients/bidirectional_rnn/fw/fw/while/Select_2_grad/tuple/control_dependency^gradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat_grad/tuple/control_dependency_1*O
_classE
CAloc:@gradients/bidirectional_rnn/fw/fw/while/Select_2_grad/Select*
N*
T0*
_output_shapes
:	@

Igradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/ConstConst^gradients/Sub_1*
value	B :*
_output_shapes
: *
dtype0

Hgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/RankConst^gradients/Sub_1*
value	B :*
_output_shapes
: *
dtype0

Ggradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/modFloorModIgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/ConstHgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/Rank*
T0*
_output_shapes
: 
Ź
Igradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/ShapeConst^gradients/Sub_1*
dtype0*
valueB"      *
_output_shapes
:
Ž
Kgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/Shape_1Const^gradients/Sub_1*
valueB"   @   *
dtype0*
_output_shapes
:
ě
Pgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/ConcatOffsetConcatOffsetGgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/modIgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/ShapeKgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/Shape_1* 
_output_shapes
::*
N

Igradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/SliceSlice\gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/tuple/control_dependencyPgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/ConcatOffsetIgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/Shape* 
_output_shapes
:
*
Index0*
T0

Kgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/Slice_1Slice\gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/tuple/control_dependencyRgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/ConcatOffset:1Kgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/Shape_1*
_output_shapes
:	@*
Index0*
T0
ö
Tgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/tuple/group_depsNoOpJ^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/SliceL^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/Slice_1

\gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/tuple/control_dependencyIdentityIgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/SliceU^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/tuple/group_deps* 
_output_shapes
:
*\
_classR
PNloc:@gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/Slice*
T0

^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/tuple/control_dependency_1IdentityKgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/Slice_1U^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/tuple/group_deps*^
_classT
RPloc:@gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/Slice_1*
T0*
_output_shapes
:	@
¨
Ogradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul/Enter_grad/b_accConst*
valueB
Ŕ*    * 
_output_shapes
:
Ŕ*
dtype0
Ë
Qgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_1EnterOgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul/Enter_grad/b_acc*
T0* 
_output_shapes
:
Ŕ*
parallel_iterations *E

frame_name75gradients/bidirectional_rnn/bw/bw/while/while_context*
is_constant( 
ź
Qgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_2MergeQgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_1Wgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul/Enter_grad/NextIteration*
N*
T0*"
_output_shapes
:
Ŕ: 
ů
Pgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul/Enter_grad/SwitchSwitchQgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_2gradients/b_count_6*
T0*,
_output_shapes
:
Ŕ:
Ŕ
ł
Mgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul/Enter_grad/AddAddRgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul/Enter_grad/Switch:1^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/tuple/control_dependency_1* 
_output_shapes
:
Ŕ*
T0
â
Wgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul/Enter_grad/NextIterationNextIterationMgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul/Enter_grad/Add*
T0* 
_output_shapes
:
Ŕ
Ö
Qgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_3ExitPgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul/Enter_grad/Switch* 
_output_shapes
:
Ŕ*
T0

Lgradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_accConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ť
Ngradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_acc_1EnterLgradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_acc*
parallel_iterations *
T0*E

frame_name75gradients/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
: *
is_constant( 
Š
Ngradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_acc_2MergeNgradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Tgradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/NextIteration*
_output_shapes
: : *
N*
T0
ß
Mgradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/SwitchSwitchNgradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_acc_2gradients/b_count_2*
_output_shapes
: : *
T0
§
Jgradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/AddAddOgradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/Switch:1bgradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
Ň
Tgradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/NextIterationNextIterationJgradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/Add*
T0*
_output_shapes
: 
Ć
Ngradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_acc_3ExitMgradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/Switch*
T0*
_output_shapes
: 

Egradients/bidirectional_rnn/fw/fw/while/Switch_4_grad_1/NextIterationNextIterationgradients/AddN_6*
_output_shapes
:	@*
T0
Ř
`gradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3fgradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enterhgradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1^gradients/Sub_1*H
_class>
<:loc:@bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter*
source	gradients*
_output_shapes

:: 
ú
fgradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnter%bidirectional_rnn/bw/bw/TensorArray_1*H
_class>
<:loc:@bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter*E

frame_name75gradients/bidirectional_rnn/bw/bw/while/while_context*
parallel_iterations *
T0*
_output_shapes
:*
is_constant(
Ľ
hgradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1EnterRbidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*E

frame_name75gradients/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
: *H
_class>
<:loc:@bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter*
is_constant(*
T0*
parallel_iterations 
 
\gradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flowIdentityhgradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1a^gradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3*
_output_shapes
: *H
_class>
<:loc:@bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter*
T0
Ś
bgradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3`gradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3mgradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2\gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/tuple/control_dependency\gradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flow*
T0*
_output_shapes
: 
Ě
gradients/AddN_7AddNNgradients/bidirectional_rnn/bw/bw/while/Select_2_grad/tuple/control_dependency^gradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat_grad/tuple/control_dependency_1*
N*
_output_shapes
:	@*O
_classE
CAloc:@gradients/bidirectional_rnn/bw/bw/while/Select_2_grad/Select*
T0
˙
gradients/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3%bidirectional_rnn/fw/fw/TensorArray_1Ngradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*8
_class.
,*loc:@bidirectional_rnn/fw/fw/TensorArray_1*
_output_shapes

:: *
source	gradients
˝
gradients/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flowIdentityNgradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_acc_3^gradients/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3*
_output_shapes
: *
T0*8
_class.
,*loc:@bidirectional_rnn/fw/fw/TensorArray_1

ugradients/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3TensorArrayGatherV3gradients/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV30bidirectional_rnn/fw/fw/TensorArrayUnstack/rangegradients/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flow*
element_shape:*-
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0
Ă
rgradients/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_depsNoOpv^gradients/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3O^gradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_acc_3
ľ
zgradients/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependencyIdentityugradients/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3s^gradients/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*
_class~
|zloc:@gradients/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3*-
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ń
|gradients/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependency_1IdentityNgradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_acc_3s^gradients/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*
T0*
_output_shapes
: *a
_classW
USloc:@gradients/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_acc_3

Lgradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_accConst*
_output_shapes
: *
dtype0*
valueB
 *    
ť
Ngradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_acc_1EnterLgradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_acc*
T0*
_output_shapes
: *
is_constant( *E

frame_name75gradients/bidirectional_rnn/bw/bw/while/while_context*
parallel_iterations 
Š
Ngradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_acc_2MergeNgradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Tgradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/NextIteration*
N*
_output_shapes
: : *
T0
ß
Mgradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/SwitchSwitchNgradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_acc_2gradients/b_count_6*
_output_shapes
: : *
T0
§
Jgradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/AddAddOgradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/Switch:1bgradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
Ň
Tgradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/NextIterationNextIterationJgradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/Add*
T0*
_output_shapes
: 
Ć
Ngradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_acc_3ExitMgradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/Switch*
_output_shapes
: *
T0

Egradients/bidirectional_rnn/bw/bw/while/Switch_4_grad_1/NextIterationNextIterationgradients/AddN_7*
T0*
_output_shapes
:	@

Bgradients/bidirectional_rnn/fw/fw/transpose_grad/InvertPermutationInvertPermutationbidirectional_rnn/fw/fw/concat*
_output_shapes
:*
T0
Ě
:gradients/bidirectional_rnn/fw/fw/transpose_grad/transpose	Transposezgradients/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependencyBgradients/bidirectional_rnn/fw/fw/transpose_grad/InvertPermutation*
Tperm0*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
˙
gradients/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3%bidirectional_rnn/bw/bw/TensorArray_1Ngradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*
_output_shapes

:: *
source	gradients*8
_class.
,*loc:@bidirectional_rnn/bw/bw/TensorArray_1
˝
gradients/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flowIdentityNgradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_acc_3^gradients/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3*8
_class.
,*loc:@bidirectional_rnn/bw/bw/TensorArray_1*
T0*
_output_shapes
: 

ugradients/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3TensorArrayGatherV3gradients/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV30bidirectional_rnn/bw/bw/TensorArrayUnstack/rangegradients/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flow*
dtype0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_shape:
Ă
rgradients/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_depsNoOpv^gradients/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3O^gradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_acc_3
ľ
zgradients/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependencyIdentityugradients/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3s^gradients/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*
_class~
|zloc:@gradients/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ń
|gradients/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependency_1IdentityNgradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_acc_3s^gradients/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*
_output_shapes
: *
T0*a
_classW
USloc:@gradients/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_acc_3

Bgradients/bidirectional_rnn/bw/bw/transpose_grad/InvertPermutationInvertPermutationbidirectional_rnn/bw/bw/concat*
_output_shapes
:*
T0
Ě
:gradients/bidirectional_rnn/bw/bw/transpose_grad/transpose	Transposezgradients/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependencyBgradients/bidirectional_rnn/bw/bw/transpose_grad/InvertPermutation*
Tperm0*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙

Cgradients/bidirectional_rnn/bw/ReverseSequence_grad/ReverseSequenceReverseSequence:gradients/bidirectional_rnn/bw/bw/transpose_grad/transposePlaceholder_2*-
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tlen0*
seq_dim*
	batch_dim *
T0
Š
gradients/AddN_8AddN:gradients/bidirectional_rnn/fw/fw/transpose_grad/transposeCgradients/bidirectional_rnn/bw/ReverseSequence_grad/ReverseSequence*-
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
N*M
_classC
A?loc:@gradients/bidirectional_rnn/fw/fw/transpose_grad/transpose

%gradients/embedding_lookup_grad/ShapeConst*
dtype0	*
_class
loc:@Embedding*%
valueB	"              *
_output_shapes
:
ľ
$gradients/embedding_lookup_grad/CastCast%gradients/embedding_lookup_grad/Shape*
_output_shapes
:*

SrcT0	*

DstT0*
_class
loc:@Embedding*
Truncate( 
j
$gradients/embedding_lookup_grad/SizeSizePlaceholder*
_output_shapes
: *
T0*
out_type0
p
.gradients/embedding_lookup_grad/ExpandDims/dimConst*
_output_shapes
: *
value	B : *
dtype0
ż
*gradients/embedding_lookup_grad/ExpandDims
ExpandDims$gradients/embedding_lookup_grad/Size.gradients/embedding_lookup_grad/ExpandDims/dim*
_output_shapes
:*
T0*

Tdim0
}
3gradients/embedding_lookup_grad/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:

5gradients/embedding_lookup_grad/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 

5gradients/embedding_lookup_grad/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:

-gradients/embedding_lookup_grad/strided_sliceStridedSlice$gradients/embedding_lookup_grad/Cast3gradients/embedding_lookup_grad/strided_slice/stack5gradients/embedding_lookup_grad/strided_slice/stack_15gradients/embedding_lookup_grad/strided_slice/stack_2*
shrink_axis_mask *
new_axis_mask *

begin_mask *
end_mask*
ellipsis_mask *
_output_shapes
:*
T0*
Index0
m
+gradients/embedding_lookup_grad/concat/axisConst*
value	B : *
_output_shapes
: *
dtype0
ô
&gradients/embedding_lookup_grad/concatConcatV2*gradients/embedding_lookup_grad/ExpandDims-gradients/embedding_lookup_grad/strided_slice+gradients/embedding_lookup_grad/concat/axis*

Tidx0*
N*
T0*
_output_shapes
:
­
'gradients/embedding_lookup_grad/ReshapeReshapegradients/AddN_8&gradients/embedding_lookup_grad/concat*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
Š
)gradients/embedding_lookup_grad/Reshape_1ReshapePlaceholder*gradients/embedding_lookup_grad/ExpandDims*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0
w
beta1_power/initial_valueConst*
dtype0*
valueB
 *fff?*
_output_shapes
: *
_class
	loc:@Bias

beta1_power
VariableV2*
_output_shapes
: *
_class
	loc:@Bias*
shape: *
shared_name *
	container *
dtype0
§
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
_class
	loc:@Bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
c
beta1_power/readIdentitybeta1_power*
_class
	loc:@Bias*
T0*
_output_shapes
: 
w
beta2_power/initial_valueConst*
valueB
 *wž?*
dtype0*
_class
	loc:@Bias*
_output_shapes
: 

beta2_power
VariableV2*
shared_name *
shape: *
_output_shapes
: *
	container *
_class
	loc:@Bias*
dtype0
§
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
_class
	loc:@Bias*
use_locking(*
validate_shape(*
_output_shapes
: *
T0
c
beta2_power/readIdentitybeta2_power*
_class
	loc:@Bias*
T0*
_output_shapes
: 

0Embedding/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@Embedding*
_output_shapes
:*
valueB"      *
dtype0

&Embedding/Adam/Initializer/zeros/ConstConst*
_class
loc:@Embedding*
dtype0*
valueB
 *    *
_output_shapes
: 
Ü
 Embedding/Adam/Initializer/zerosFill0Embedding/Adam/Initializer/zeros/shape_as_tensor&Embedding/Adam/Initializer/zeros/Const*
_output_shapes
:	*
_class
loc:@Embedding*
T0*

index_type0
˘
Embedding/Adam
VariableV2*
_output_shapes
:	*
shape:	*
dtype0*
	container *
_class
loc:@Embedding*
shared_name 
Â
Embedding/Adam/AssignAssignEmbedding/Adam Embedding/Adam/Initializer/zeros*
T0*
_class
loc:@Embedding*
use_locking(*
validate_shape(*
_output_shapes
:	
w
Embedding/Adam/readIdentityEmbedding/Adam*
_class
loc:@Embedding*
T0*
_output_shapes
:	
Ą
2Embedding/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_class
loc:@Embedding*
valueB"      *
_output_shapes
:

(Embedding/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *
_class
loc:@Embedding
â
"Embedding/Adam_1/Initializer/zerosFill2Embedding/Adam_1/Initializer/zeros/shape_as_tensor(Embedding/Adam_1/Initializer/zeros/Const*
_class
loc:@Embedding*
_output_shapes
:	*

index_type0*
T0
¤
Embedding/Adam_1
VariableV2*
	container *
_class
loc:@Embedding*
dtype0*
shape:	*
_output_shapes
:	*
shared_name 
Č
Embedding/Adam_1/AssignAssignEmbedding/Adam_1"Embedding/Adam_1/Initializer/zeros*
validate_shape(*
T0*
_class
loc:@Embedding*
use_locking(*
_output_shapes
:	
{
Embedding/Adam_1/readIdentityEmbedding/Adam_1*
_output_shapes
:	*
T0*
_class
loc:@Embedding
ă
Rbidirectional_rnn/fw/basic_lstm_cell/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*>
_class4
20loc:@bidirectional_rnn/fw/basic_lstm_cell/kernel*
valueB"@     
Í
Hbidirectional_rnn/fw/basic_lstm_cell/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *>
_class4
20loc:@bidirectional_rnn/fw/basic_lstm_cell/kernel
ĺ
Bbidirectional_rnn/fw/basic_lstm_cell/kernel/Adam/Initializer/zerosFillRbidirectional_rnn/fw/basic_lstm_cell/kernel/Adam/Initializer/zeros/shape_as_tensorHbidirectional_rnn/fw/basic_lstm_cell/kernel/Adam/Initializer/zeros/Const*>
_class4
20loc:@bidirectional_rnn/fw/basic_lstm_cell/kernel*
T0* 
_output_shapes
:
Ŕ*

index_type0
č
0bidirectional_rnn/fw/basic_lstm_cell/kernel/Adam
VariableV2* 
_output_shapes
:
Ŕ*
shared_name *>
_class4
20loc:@bidirectional_rnn/fw/basic_lstm_cell/kernel*
shape:
Ŕ*
	container *
dtype0
Ë
7bidirectional_rnn/fw/basic_lstm_cell/kernel/Adam/AssignAssign0bidirectional_rnn/fw/basic_lstm_cell/kernel/AdamBbidirectional_rnn/fw/basic_lstm_cell/kernel/Adam/Initializer/zeros*>
_class4
20loc:@bidirectional_rnn/fw/basic_lstm_cell/kernel*
validate_shape(*
T0* 
_output_shapes
:
Ŕ*
use_locking(
Ţ
5bidirectional_rnn/fw/basic_lstm_cell/kernel/Adam/readIdentity0bidirectional_rnn/fw/basic_lstm_cell/kernel/Adam*>
_class4
20loc:@bidirectional_rnn/fw/basic_lstm_cell/kernel* 
_output_shapes
:
Ŕ*
T0
ĺ
Tbidirectional_rnn/fw/basic_lstm_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*>
_class4
20loc:@bidirectional_rnn/fw/basic_lstm_cell/kernel*
valueB"@     *
dtype0*
_output_shapes
:
Ď
Jbidirectional_rnn/fw/basic_lstm_cell/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*>
_class4
20loc:@bidirectional_rnn/fw/basic_lstm_cell/kernel*
_output_shapes
: *
valueB
 *    
ë
Dbidirectional_rnn/fw/basic_lstm_cell/kernel/Adam_1/Initializer/zerosFillTbidirectional_rnn/fw/basic_lstm_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensorJbidirectional_rnn/fw/basic_lstm_cell/kernel/Adam_1/Initializer/zeros/Const*
T0* 
_output_shapes
:
Ŕ*

index_type0*>
_class4
20loc:@bidirectional_rnn/fw/basic_lstm_cell/kernel
ę
2bidirectional_rnn/fw/basic_lstm_cell/kernel/Adam_1
VariableV2*
shape:
Ŕ*
dtype0*
	container *>
_class4
20loc:@bidirectional_rnn/fw/basic_lstm_cell/kernel* 
_output_shapes
:
Ŕ*
shared_name 
Ń
9bidirectional_rnn/fw/basic_lstm_cell/kernel/Adam_1/AssignAssign2bidirectional_rnn/fw/basic_lstm_cell/kernel/Adam_1Dbidirectional_rnn/fw/basic_lstm_cell/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0* 
_output_shapes
:
Ŕ*>
_class4
20loc:@bidirectional_rnn/fw/basic_lstm_cell/kernel*
validate_shape(
â
7bidirectional_rnn/fw/basic_lstm_cell/kernel/Adam_1/readIdentity2bidirectional_rnn/fw/basic_lstm_cell/kernel/Adam_1*
T0* 
_output_shapes
:
Ŕ*>
_class4
20loc:@bidirectional_rnn/fw/basic_lstm_cell/kernel
Í
@bidirectional_rnn/fw/basic_lstm_cell/bias/Adam/Initializer/zerosConst*
_output_shapes	
:*
dtype0*
valueB*    *<
_class2
0.loc:@bidirectional_rnn/fw/basic_lstm_cell/bias
Ú
.bidirectional_rnn/fw/basic_lstm_cell/bias/Adam
VariableV2*
dtype0*<
_class2
0.loc:@bidirectional_rnn/fw/basic_lstm_cell/bias*
shape:*
_output_shapes	
:*
shared_name *
	container 
ž
5bidirectional_rnn/fw/basic_lstm_cell/bias/Adam/AssignAssign.bidirectional_rnn/fw/basic_lstm_cell/bias/Adam@bidirectional_rnn/fw/basic_lstm_cell/bias/Adam/Initializer/zeros*
validate_shape(*<
_class2
0.loc:@bidirectional_rnn/fw/basic_lstm_cell/bias*
_output_shapes	
:*
T0*
use_locking(
Ó
3bidirectional_rnn/fw/basic_lstm_cell/bias/Adam/readIdentity.bidirectional_rnn/fw/basic_lstm_cell/bias/Adam*
T0*
_output_shapes	
:*<
_class2
0.loc:@bidirectional_rnn/fw/basic_lstm_cell/bias
Ď
Bbidirectional_rnn/fw/basic_lstm_cell/bias/Adam_1/Initializer/zerosConst*
valueB*    *
dtype0*
_output_shapes	
:*<
_class2
0.loc:@bidirectional_rnn/fw/basic_lstm_cell/bias
Ü
0bidirectional_rnn/fw/basic_lstm_cell/bias/Adam_1
VariableV2*<
_class2
0.loc:@bidirectional_rnn/fw/basic_lstm_cell/bias*
shape:*
shared_name *
_output_shapes	
:*
dtype0*
	container 
Ä
7bidirectional_rnn/fw/basic_lstm_cell/bias/Adam_1/AssignAssign0bidirectional_rnn/fw/basic_lstm_cell/bias/Adam_1Bbidirectional_rnn/fw/basic_lstm_cell/bias/Adam_1/Initializer/zeros*
T0*
use_locking(*<
_class2
0.loc:@bidirectional_rnn/fw/basic_lstm_cell/bias*
validate_shape(*
_output_shapes	
:
×
5bidirectional_rnn/fw/basic_lstm_cell/bias/Adam_1/readIdentity0bidirectional_rnn/fw/basic_lstm_cell/bias/Adam_1*<
_class2
0.loc:@bidirectional_rnn/fw/basic_lstm_cell/bias*
_output_shapes	
:*
T0
ă
Rbidirectional_rnn/bw/basic_lstm_cell/kernel/Adam/Initializer/zeros/shape_as_tensorConst*>
_class4
20loc:@bidirectional_rnn/bw/basic_lstm_cell/kernel*
valueB"@     *
dtype0*
_output_shapes
:
Í
Hbidirectional_rnn/bw/basic_lstm_cell/kernel/Adam/Initializer/zeros/ConstConst*>
_class4
20loc:@bidirectional_rnn/bw/basic_lstm_cell/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
ĺ
Bbidirectional_rnn/bw/basic_lstm_cell/kernel/Adam/Initializer/zerosFillRbidirectional_rnn/bw/basic_lstm_cell/kernel/Adam/Initializer/zeros/shape_as_tensorHbidirectional_rnn/bw/basic_lstm_cell/kernel/Adam/Initializer/zeros/Const* 
_output_shapes
:
Ŕ*>
_class4
20loc:@bidirectional_rnn/bw/basic_lstm_cell/kernel*
T0*

index_type0
č
0bidirectional_rnn/bw/basic_lstm_cell/kernel/Adam
VariableV2*
shape:
Ŕ*
	container *
shared_name *>
_class4
20loc:@bidirectional_rnn/bw/basic_lstm_cell/kernel* 
_output_shapes
:
Ŕ*
dtype0
Ë
7bidirectional_rnn/bw/basic_lstm_cell/kernel/Adam/AssignAssign0bidirectional_rnn/bw/basic_lstm_cell/kernel/AdamBbidirectional_rnn/bw/basic_lstm_cell/kernel/Adam/Initializer/zeros* 
_output_shapes
:
Ŕ*
use_locking(*>
_class4
20loc:@bidirectional_rnn/bw/basic_lstm_cell/kernel*
T0*
validate_shape(
Ţ
5bidirectional_rnn/bw/basic_lstm_cell/kernel/Adam/readIdentity0bidirectional_rnn/bw/basic_lstm_cell/kernel/Adam* 
_output_shapes
:
Ŕ*>
_class4
20loc:@bidirectional_rnn/bw/basic_lstm_cell/kernel*
T0
ĺ
Tbidirectional_rnn/bw/basic_lstm_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@     *>
_class4
20loc:@bidirectional_rnn/bw/basic_lstm_cell/kernel
Ď
Jbidirectional_rnn/bw/basic_lstm_cell/kernel/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *>
_class4
20loc:@bidirectional_rnn/bw/basic_lstm_cell/kernel*
dtype0*
valueB
 *    
ë
Dbidirectional_rnn/bw/basic_lstm_cell/kernel/Adam_1/Initializer/zerosFillTbidirectional_rnn/bw/basic_lstm_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensorJbidirectional_rnn/bw/basic_lstm_cell/kernel/Adam_1/Initializer/zeros/Const*
T0*>
_class4
20loc:@bidirectional_rnn/bw/basic_lstm_cell/kernel*

index_type0* 
_output_shapes
:
Ŕ
ę
2bidirectional_rnn/bw/basic_lstm_cell/kernel/Adam_1
VariableV2* 
_output_shapes
:
Ŕ*
dtype0*
	container *
shared_name *
shape:
Ŕ*>
_class4
20loc:@bidirectional_rnn/bw/basic_lstm_cell/kernel
Ń
9bidirectional_rnn/bw/basic_lstm_cell/kernel/Adam_1/AssignAssign2bidirectional_rnn/bw/basic_lstm_cell/kernel/Adam_1Dbidirectional_rnn/bw/basic_lstm_cell/kernel/Adam_1/Initializer/zeros*
validate_shape(* 
_output_shapes
:
Ŕ*
use_locking(*
T0*>
_class4
20loc:@bidirectional_rnn/bw/basic_lstm_cell/kernel
â
7bidirectional_rnn/bw/basic_lstm_cell/kernel/Adam_1/readIdentity2bidirectional_rnn/bw/basic_lstm_cell/kernel/Adam_1*>
_class4
20loc:@bidirectional_rnn/bw/basic_lstm_cell/kernel* 
_output_shapes
:
Ŕ*
T0
Í
@bidirectional_rnn/bw/basic_lstm_cell/bias/Adam/Initializer/zerosConst*
_output_shapes	
:*
valueB*    *<
_class2
0.loc:@bidirectional_rnn/bw/basic_lstm_cell/bias*
dtype0
Ú
.bidirectional_rnn/bw/basic_lstm_cell/bias/Adam
VariableV2*
dtype0*
	container *<
_class2
0.loc:@bidirectional_rnn/bw/basic_lstm_cell/bias*
shape:*
shared_name *
_output_shapes	
:
ž
5bidirectional_rnn/bw/basic_lstm_cell/bias/Adam/AssignAssign.bidirectional_rnn/bw/basic_lstm_cell/bias/Adam@bidirectional_rnn/bw/basic_lstm_cell/bias/Adam/Initializer/zeros*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*<
_class2
0.loc:@bidirectional_rnn/bw/basic_lstm_cell/bias
Ó
3bidirectional_rnn/bw/basic_lstm_cell/bias/Adam/readIdentity.bidirectional_rnn/bw/basic_lstm_cell/bias/Adam*
_output_shapes	
:*<
_class2
0.loc:@bidirectional_rnn/bw/basic_lstm_cell/bias*
T0
Ď
Bbidirectional_rnn/bw/basic_lstm_cell/bias/Adam_1/Initializer/zerosConst*<
_class2
0.loc:@bidirectional_rnn/bw/basic_lstm_cell/bias*
valueB*    *
_output_shapes	
:*
dtype0
Ü
0bidirectional_rnn/bw/basic_lstm_cell/bias/Adam_1
VariableV2*
dtype0*
	container *<
_class2
0.loc:@bidirectional_rnn/bw/basic_lstm_cell/bias*
shape:*
shared_name *
_output_shapes	
:
Ä
7bidirectional_rnn/bw/basic_lstm_cell/bias/Adam_1/AssignAssign0bidirectional_rnn/bw/basic_lstm_cell/bias/Adam_1Bbidirectional_rnn/bw/basic_lstm_cell/bias/Adam_1/Initializer/zeros*
use_locking(*
_output_shapes	
:*
T0*
validate_shape(*<
_class2
0.loc:@bidirectional_rnn/bw/basic_lstm_cell/bias
×
5bidirectional_rnn/bw/basic_lstm_cell/bias/Adam_1/readIdentity0bidirectional_rnn/bw/basic_lstm_cell/bias/Adam_1*
_output_shapes	
:*<
_class2
0.loc:@bidirectional_rnn/bw/basic_lstm_cell/bias*
T0

Weights/Adam/Initializer/zerosConst*
dtype0*
valueB	*    *
_class
loc:@Weights*
_output_shapes
:	

Weights/Adam
VariableV2*
shape:	*
dtype0*
_class
loc:@Weights*
	container *
shared_name *
_output_shapes
:	
ş
Weights/Adam/AssignAssignWeights/AdamWeights/Adam/Initializer/zeros*
_class
loc:@Weights*
use_locking(*
T0*
validate_shape(*
_output_shapes
:	
q
Weights/Adam/readIdentityWeights/Adam*
_output_shapes
:	*
T0*
_class
loc:@Weights

 Weights/Adam_1/Initializer/zerosConst*
_output_shapes
:	*
valueB	*    *
dtype0*
_class
loc:@Weights
 
Weights/Adam_1
VariableV2*
shared_name *
shape:	*
	container *
_output_shapes
:	*
dtype0*
_class
loc:@Weights
Ŕ
Weights/Adam_1/AssignAssignWeights/Adam_1 Weights/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
T0*
_class
loc:@Weights*
_output_shapes
:	
u
Weights/Adam_1/readIdentityWeights/Adam_1*
T0*
_output_shapes
:	*
_class
loc:@Weights

Bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    *
_class
	loc:@Bias

	Bias/Adam
VariableV2*
shape:*
_output_shapes
:*
shared_name *
_class
	loc:@Bias*
dtype0*
	container 
Š
Bias/Adam/AssignAssign	Bias/AdamBias/Adam/Initializer/zeros*
use_locking(*
_output_shapes
:*
validate_shape(*
_class
	loc:@Bias*
T0
c
Bias/Adam/readIdentity	Bias/Adam*
_class
	loc:@Bias*
_output_shapes
:*
T0

Bias/Adam_1/Initializer/zerosConst*
dtype0*
valueB*    *
_output_shapes
:*
_class
	loc:@Bias

Bias/Adam_1
VariableV2*
shared_name *
	container *
shape:*
dtype0*
_class
	loc:@Bias*
_output_shapes
:
Ż
Bias/Adam_1/AssignAssignBias/Adam_1Bias/Adam_1/Initializer/zeros*
validate_shape(*
_class
	loc:@Bias*
use_locking(*
T0*
_output_shapes
:
g
Bias/Adam_1/readIdentityBias/Adam_1*
_class
	loc:@Bias*
_output_shapes
:*
T0
W
Adam/learning_rateConst*
dtype0*
valueB
 *o:*
_output_shapes
: 
O

Adam/beta1Const*
_output_shapes
: *
valueB
 *fff?*
dtype0
O

Adam/beta2Const*
dtype0*
_output_shapes
: *
valueB
 *wž?
Q
Adam/epsilonConst*
_output_shapes
: *
valueB
 *wĚ+2*
dtype0
ť
Adam/update_Embedding/UniqueUnique)gradients/embedding_lookup_grad/Reshape_1*
_class
loc:@Embedding*
out_idx0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

Adam/update_Embedding/ShapeShapeAdam/update_Embedding/Unique*
_output_shapes
:*
_class
loc:@Embedding*
T0*
out_type0

)Adam/update_Embedding/strided_slice/stackConst*
_class
loc:@Embedding*
dtype0*
_output_shapes
:*
valueB: 

+Adam/update_Embedding/strided_slice/stack_1Const*
dtype0*
_class
loc:@Embedding*
_output_shapes
:*
valueB:

+Adam/update_Embedding/strided_slice/stack_2Const*
valueB:*
dtype0*
_class
loc:@Embedding*
_output_shapes
:

#Adam/update_Embedding/strided_sliceStridedSliceAdam/update_Embedding/Shape)Adam/update_Embedding/strided_slice/stack+Adam/update_Embedding/strided_slice/stack_1+Adam/update_Embedding/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
end_mask *

begin_mask *
ellipsis_mask *
_output_shapes
: *
_class
loc:@Embedding*
new_axis_mask 
Ą
(Adam/update_Embedding/UnsortedSegmentSumUnsortedSegmentSum'gradients/embedding_lookup_grad/ReshapeAdam/update_Embedding/Unique:1#Adam/update_Embedding/strided_slice*
Tnumsegments0*
Tindices0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
_class
loc:@Embedding
~
Adam/update_Embedding/sub/xConst*
_class
loc:@Embedding*
valueB
 *  ?*
_output_shapes
: *
dtype0

Adam/update_Embedding/subSubAdam/update_Embedding/sub/xbeta2_power/read*
T0*
_class
loc:@Embedding*
_output_shapes
: 
|
Adam/update_Embedding/SqrtSqrtAdam/update_Embedding/sub*
_class
loc:@Embedding*
T0*
_output_shapes
: 

Adam/update_Embedding/mulMulAdam/learning_rateAdam/update_Embedding/Sqrt*
T0*
_class
loc:@Embedding*
_output_shapes
: 

Adam/update_Embedding/sub_1/xConst*
valueB
 *  ?*
dtype0*
_class
loc:@Embedding*
_output_shapes
: 

Adam/update_Embedding/sub_1SubAdam/update_Embedding/sub_1/xbeta1_power/read*
T0*
_class
loc:@Embedding*
_output_shapes
: 

Adam/update_Embedding/truedivRealDivAdam/update_Embedding/mulAdam/update_Embedding/sub_1*
T0*
_class
loc:@Embedding*
_output_shapes
: 

Adam/update_Embedding/sub_2/xConst*
valueB
 *  ?*
_output_shapes
: *
_class
loc:@Embedding*
dtype0

Adam/update_Embedding/sub_2SubAdam/update_Embedding/sub_2/x
Adam/beta1*
_class
loc:@Embedding*
T0*
_output_shapes
: 
ş
Adam/update_Embedding/mul_1Mul(Adam/update_Embedding/UnsortedSegmentSumAdam/update_Embedding/sub_2*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
_class
loc:@Embedding

Adam/update_Embedding/mul_2MulEmbedding/Adam/read
Adam/beta1*
T0*
_output_shapes
:	*
_class
loc:@Embedding
Ä
Adam/update_Embedding/AssignAssignEmbedding/AdamAdam/update_Embedding/mul_2*
T0*
_class
loc:@Embedding*
validate_shape(*
_output_shapes
:	*
use_locking( 

 Adam/update_Embedding/ScatterAdd
ScatterAddEmbedding/AdamAdam/update_Embedding/UniqueAdam/update_Embedding/mul_1^Adam/update_Embedding/Assign*
use_locking( *
T0*
Tindices0*
_output_shapes
:	*
_class
loc:@Embedding
Ç
Adam/update_Embedding/mul_3Mul(Adam/update_Embedding/UnsortedSegmentSum(Adam/update_Embedding/UnsortedSegmentSum*
_class
loc:@Embedding*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Adam/update_Embedding/sub_3/xConst*
_class
loc:@Embedding*
_output_shapes
: *
dtype0*
valueB
 *  ?

Adam/update_Embedding/sub_3SubAdam/update_Embedding/sub_3/x
Adam/beta2*
_output_shapes
: *
T0*
_class
loc:@Embedding
­
Adam/update_Embedding/mul_4MulAdam/update_Embedding/mul_3Adam/update_Embedding/sub_3*
_class
loc:@Embedding*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

Adam/update_Embedding/mul_5MulEmbedding/Adam_1/read
Adam/beta2*
T0*
_output_shapes
:	*
_class
loc:@Embedding
Č
Adam/update_Embedding/Assign_1AssignEmbedding/Adam_1Adam/update_Embedding/mul_5*
_output_shapes
:	*
use_locking( *
_class
loc:@Embedding*
T0*
validate_shape(

"Adam/update_Embedding/ScatterAdd_1
ScatterAddEmbedding/Adam_1Adam/update_Embedding/UniqueAdam/update_Embedding/mul_4^Adam/update_Embedding/Assign_1*
T0*
_output_shapes
:	*
Tindices0*
use_locking( *
_class
loc:@Embedding

Adam/update_Embedding/Sqrt_1Sqrt"Adam/update_Embedding/ScatterAdd_1*
_output_shapes
:	*
T0*
_class
loc:@Embedding
Ť
Adam/update_Embedding/mul_6MulAdam/update_Embedding/truediv Adam/update_Embedding/ScatterAdd*
T0*
_class
loc:@Embedding*
_output_shapes
:	

Adam/update_Embedding/addAddAdam/update_Embedding/Sqrt_1Adam/epsilon*
_output_shapes
:	*
T0*
_class
loc:@Embedding
Ş
Adam/update_Embedding/truediv_1RealDivAdam/update_Embedding/mul_6Adam/update_Embedding/add*
_class
loc:@Embedding*
T0*
_output_shapes
:	
ł
Adam/update_Embedding/AssignSub	AssignSub	EmbeddingAdam/update_Embedding/truediv_1*
_output_shapes
:	*
T0*
_class
loc:@Embedding*
use_locking( 
°
 Adam/update_Embedding/group_depsNoOp ^Adam/update_Embedding/AssignSub!^Adam/update_Embedding/ScatterAdd#^Adam/update_Embedding/ScatterAdd_1*
_class
loc:@Embedding
¤
AAdam/update_bidirectional_rnn/fw/basic_lstm_cell/kernel/ApplyAdam	ApplyAdam+bidirectional_rnn/fw/basic_lstm_cell/kernel0bidirectional_rnn/fw/basic_lstm_cell/kernel/Adam2bidirectional_rnn/fw/basic_lstm_cell/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonQgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_3* 
_output_shapes
:
Ŕ*>
_class4
20loc:@bidirectional_rnn/fw/basic_lstm_cell/kernel*
use_locking( *
use_nesterov( *
T0

?Adam/update_bidirectional_rnn/fw/basic_lstm_cell/bias/ApplyAdam	ApplyAdam)bidirectional_rnn/fw/basic_lstm_cell/bias.bidirectional_rnn/fw/basic_lstm_cell/bias/Adam0bidirectional_rnn/fw/basic_lstm_cell/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonRgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3*<
_class2
0.loc:@bidirectional_rnn/fw/basic_lstm_cell/bias*
use_nesterov( *
T0*
use_locking( *
_output_shapes	
:
¤
AAdam/update_bidirectional_rnn/bw/basic_lstm_cell/kernel/ApplyAdam	ApplyAdam+bidirectional_rnn/bw/basic_lstm_cell/kernel0bidirectional_rnn/bw/basic_lstm_cell/kernel/Adam2bidirectional_rnn/bw/basic_lstm_cell/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonQgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_3*>
_class4
20loc:@bidirectional_rnn/bw/basic_lstm_cell/kernel*
use_locking( *
use_nesterov( *
T0* 
_output_shapes
:
Ŕ

?Adam/update_bidirectional_rnn/bw/basic_lstm_cell/bias/ApplyAdam	ApplyAdam)bidirectional_rnn/bw/basic_lstm_cell/bias.bidirectional_rnn/bw/basic_lstm_cell/bias/Adam0bidirectional_rnn/bw/basic_lstm_cell/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonRgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3*
use_nesterov( *
T0*
_output_shapes	
:*
use_locking( *<
_class2
0.loc:@bidirectional_rnn/bw/basic_lstm_cell/bias
Î
Adam/update_Weights/ApplyAdam	ApplyAdamWeightsWeights/AdamWeights/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/MatMul_grad/tuple/control_dependency_1*
use_locking( *
use_nesterov( *
T0*
_class
loc:@Weights*
_output_shapes
:	
ˇ
Adam/update_Bias/ApplyAdam	ApplyAdamBias	Bias/AdamBias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon-gradients/add_grad/tuple/control_dependency_1*
use_nesterov( *
use_locking( *
T0*
_output_shapes
:*
_class
	loc:@Bias
Ó
Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_Bias/ApplyAdam!^Adam/update_Embedding/group_deps^Adam/update_Weights/ApplyAdam@^Adam/update_bidirectional_rnn/bw/basic_lstm_cell/bias/ApplyAdamB^Adam/update_bidirectional_rnn/bw/basic_lstm_cell/kernel/ApplyAdam@^Adam/update_bidirectional_rnn/fw/basic_lstm_cell/bias/ApplyAdamB^Adam/update_bidirectional_rnn/fw/basic_lstm_cell/kernel/ApplyAdam*
_class
	loc:@Bias*
_output_shapes
: *
T0

Adam/AssignAssignbeta1_powerAdam/mul*
T0*
validate_shape(*
use_locking( *
_output_shapes
: *
_class
	loc:@Bias
Ő

Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_Bias/ApplyAdam!^Adam/update_Embedding/group_deps^Adam/update_Weights/ApplyAdam@^Adam/update_bidirectional_rnn/bw/basic_lstm_cell/bias/ApplyAdamB^Adam/update_bidirectional_rnn/bw/basic_lstm_cell/kernel/ApplyAdam@^Adam/update_bidirectional_rnn/fw/basic_lstm_cell/bias/ApplyAdamB^Adam/update_bidirectional_rnn/fw/basic_lstm_cell/kernel/ApplyAdam*
_class
	loc:@Bias*
_output_shapes
: *
T0

Adam/Assign_1Assignbeta2_power
Adam/mul_1*
_class
	loc:@Bias*
T0*
validate_shape(*
use_locking( *
_output_shapes
: 

AdamNoOp^Adam/Assign^Adam/Assign_1^Adam/update_Bias/ApplyAdam!^Adam/update_Embedding/group_deps^Adam/update_Weights/ApplyAdam@^Adam/update_bidirectional_rnn/bw/basic_lstm_cell/bias/ApplyAdamB^Adam/update_bidirectional_rnn/bw/basic_lstm_cell/kernel/ApplyAdam@^Adam/update_bidirectional_rnn/fw/basic_lstm_cell/bias/ApplyAdamB^Adam/update_bidirectional_rnn/fw/basic_lstm_cell/kernel/ApplyAdam

initNoOp^Bias/Adam/Assign^Bias/Adam_1/Assign^Bias/Assign^Embedding/Adam/Assign^Embedding/Adam_1/Assign^Embedding/Assign^Weights/Adam/Assign^Weights/Adam_1/Assign^Weights/Assign^beta1_power/Assign^beta2_power/Assign6^bidirectional_rnn/bw/basic_lstm_cell/bias/Adam/Assign8^bidirectional_rnn/bw/basic_lstm_cell/bias/Adam_1/Assign1^bidirectional_rnn/bw/basic_lstm_cell/bias/Assign8^bidirectional_rnn/bw/basic_lstm_cell/kernel/Adam/Assign:^bidirectional_rnn/bw/basic_lstm_cell/kernel/Adam_1/Assign3^bidirectional_rnn/bw/basic_lstm_cell/kernel/Assign6^bidirectional_rnn/fw/basic_lstm_cell/bias/Adam/Assign8^bidirectional_rnn/fw/basic_lstm_cell/bias/Adam_1/Assign1^bidirectional_rnn/fw/basic_lstm_cell/bias/Assign8^bidirectional_rnn/fw/basic_lstm_cell/kernel/Adam/Assign:^bidirectional_rnn/fw/basic_lstm_cell/kernel/Adam_1/Assign3^bidirectional_rnn/fw/basic_lstm_cell/kernel/Assign
Y
save/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
n
save/filenamePlaceholderWithDefaultsave/filename/input*
dtype0*
_output_shapes
: *
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
shape: *
_output_shapes
: *
dtype0
Ź
save/SaveV2/tensor_namesConst*
_output_shapes
:*ß
valueŐBŇBBiasB	EmbeddingBWeightsB)bidirectional_rnn/bw/basic_lstm_cell/biasB+bidirectional_rnn/bw/basic_lstm_cell/kernelB)bidirectional_rnn/fw/basic_lstm_cell/biasB+bidirectional_rnn/fw/basic_lstm_cell/kernel*
dtype0
q
save/SaveV2/shape_and_slicesConst*
dtype0*!
valueBB B B B B B B *
_output_shapes
:
ş
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesBias	EmbeddingWeights)bidirectional_rnn/bw/basic_lstm_cell/bias+bidirectional_rnn/bw/basic_lstm_cell/kernel)bidirectional_rnn/fw/basic_lstm_cell/bias+bidirectional_rnn/fw/basic_lstm_cell/kernel*
dtypes
	2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
ž
save/RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ß
valueŐBŇBBiasB	EmbeddingBWeightsB)bidirectional_rnn/bw/basic_lstm_cell/biasB+bidirectional_rnn/bw/basic_lstm_cell/kernelB)bidirectional_rnn/fw/basic_lstm_cell/biasB+bidirectional_rnn/fw/basic_lstm_cell/kernel

save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 
˝
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
	2*0
_output_shapes
:::::::

save/AssignAssignBiassave/RestoreV2*
T0*
validate_shape(*
_output_shapes
:*
use_locking(*
_class
	loc:@Bias
Ľ
save/Assign_1Assign	Embeddingsave/RestoreV2:1*
_output_shapes
:	*
_class
loc:@Embedding*
T0*
validate_shape(*
use_locking(
Ą
save/Assign_2AssignWeightssave/RestoreV2:2*
_output_shapes
:	*
validate_shape(*
T0*
use_locking(*
_class
loc:@Weights
á
save/Assign_3Assign)bidirectional_rnn/bw/basic_lstm_cell/biassave/RestoreV2:3*
_output_shapes	
:*
use_locking(*
T0*
validate_shape(*<
_class2
0.loc:@bidirectional_rnn/bw/basic_lstm_cell/bias
ę
save/Assign_4Assign+bidirectional_rnn/bw/basic_lstm_cell/kernelsave/RestoreV2:4*
T0* 
_output_shapes
:
Ŕ*
use_locking(*
validate_shape(*>
_class4
20loc:@bidirectional_rnn/bw/basic_lstm_cell/kernel
á
save/Assign_5Assign)bidirectional_rnn/fw/basic_lstm_cell/biassave/RestoreV2:5*
_output_shapes	
:*
validate_shape(*
use_locking(*
T0*<
_class2
0.loc:@bidirectional_rnn/fw/basic_lstm_cell/bias
ę
save/Assign_6Assign+bidirectional_rnn/fw/basic_lstm_cell/kernelsave/RestoreV2:6*>
_class4
20loc:@bidirectional_rnn/fw/basic_lstm_cell/kernel*
use_locking(*
T0* 
_output_shapes
:
Ŕ*
validate_shape(

save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6

init_1NoOp^Bias/Adam/Assign^Bias/Adam_1/Assign^Bias/Assign^Embedding/Adam/Assign^Embedding/Adam_1/Assign^Embedding/Assign^Weights/Adam/Assign^Weights/Adam_1/Assign^Weights/Assign^beta1_power/Assign^beta2_power/Assign6^bidirectional_rnn/bw/basic_lstm_cell/bias/Adam/Assign8^bidirectional_rnn/bw/basic_lstm_cell/bias/Adam_1/Assign1^bidirectional_rnn/bw/basic_lstm_cell/bias/Assign8^bidirectional_rnn/bw/basic_lstm_cell/kernel/Adam/Assign:^bidirectional_rnn/bw/basic_lstm_cell/kernel/Adam_1/Assign3^bidirectional_rnn/bw/basic_lstm_cell/kernel/Assign6^bidirectional_rnn/fw/basic_lstm_cell/bias/Adam/Assign8^bidirectional_rnn/fw/basic_lstm_cell/bias/Adam_1/Assign1^bidirectional_rnn/fw/basic_lstm_cell/bias/Assign8^bidirectional_rnn/fw/basic_lstm_cell/kernel/Adam/Assign:^bidirectional_rnn/fw/basic_lstm_cell/kernel/Adam_1/Assign3^bidirectional_rnn/fw/basic_lstm_cell/kernel/Assign"&"ę
	variablesÜŮ
[
Embedding:0Embedding/AssignEmbedding/read:02&Embedding/Initializer/random_uniform:08
ă
-bidirectional_rnn/fw/basic_lstm_cell/kernel:02bidirectional_rnn/fw/basic_lstm_cell/kernel/Assign2bidirectional_rnn/fw/basic_lstm_cell/kernel/read:02Hbidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform:08
Ň
+bidirectional_rnn/fw/basic_lstm_cell/bias:00bidirectional_rnn/fw/basic_lstm_cell/bias/Assign0bidirectional_rnn/fw/basic_lstm_cell/bias/read:02=bidirectional_rnn/fw/basic_lstm_cell/bias/Initializer/zeros:08
ă
-bidirectional_rnn/bw/basic_lstm_cell/kernel:02bidirectional_rnn/bw/basic_lstm_cell/kernel/Assign2bidirectional_rnn/bw/basic_lstm_cell/kernel/read:02Hbidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform:08
Ň
+bidirectional_rnn/bw/basic_lstm_cell/bias:00bidirectional_rnn/bw/basic_lstm_cell/bias/Assign0bidirectional_rnn/bw/basic_lstm_cell/bias/read:02=bidirectional_rnn/bw/basic_lstm_cell/bias/Initializer/zeros:08
S
	Weights:0Weights/AssignWeights/read:02$Weights/Initializer/random_uniform:08
G
Bias:0Bias/AssignBias/read:02!Bias/Initializer/random_uniform:08
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
d
Embedding/Adam:0Embedding/Adam/AssignEmbedding/Adam/read:02"Embedding/Adam/Initializer/zeros:0
l
Embedding/Adam_1:0Embedding/Adam_1/AssignEmbedding/Adam_1/read:02$Embedding/Adam_1/Initializer/zeros:0
ě
2bidirectional_rnn/fw/basic_lstm_cell/kernel/Adam:07bidirectional_rnn/fw/basic_lstm_cell/kernel/Adam/Assign7bidirectional_rnn/fw/basic_lstm_cell/kernel/Adam/read:02Dbidirectional_rnn/fw/basic_lstm_cell/kernel/Adam/Initializer/zeros:0
ô
4bidirectional_rnn/fw/basic_lstm_cell/kernel/Adam_1:09bidirectional_rnn/fw/basic_lstm_cell/kernel/Adam_1/Assign9bidirectional_rnn/fw/basic_lstm_cell/kernel/Adam_1/read:02Fbidirectional_rnn/fw/basic_lstm_cell/kernel/Adam_1/Initializer/zeros:0
ä
0bidirectional_rnn/fw/basic_lstm_cell/bias/Adam:05bidirectional_rnn/fw/basic_lstm_cell/bias/Adam/Assign5bidirectional_rnn/fw/basic_lstm_cell/bias/Adam/read:02Bbidirectional_rnn/fw/basic_lstm_cell/bias/Adam/Initializer/zeros:0
ě
2bidirectional_rnn/fw/basic_lstm_cell/bias/Adam_1:07bidirectional_rnn/fw/basic_lstm_cell/bias/Adam_1/Assign7bidirectional_rnn/fw/basic_lstm_cell/bias/Adam_1/read:02Dbidirectional_rnn/fw/basic_lstm_cell/bias/Adam_1/Initializer/zeros:0
ě
2bidirectional_rnn/bw/basic_lstm_cell/kernel/Adam:07bidirectional_rnn/bw/basic_lstm_cell/kernel/Adam/Assign7bidirectional_rnn/bw/basic_lstm_cell/kernel/Adam/read:02Dbidirectional_rnn/bw/basic_lstm_cell/kernel/Adam/Initializer/zeros:0
ô
4bidirectional_rnn/bw/basic_lstm_cell/kernel/Adam_1:09bidirectional_rnn/bw/basic_lstm_cell/kernel/Adam_1/Assign9bidirectional_rnn/bw/basic_lstm_cell/kernel/Adam_1/read:02Fbidirectional_rnn/bw/basic_lstm_cell/kernel/Adam_1/Initializer/zeros:0
ä
0bidirectional_rnn/bw/basic_lstm_cell/bias/Adam:05bidirectional_rnn/bw/basic_lstm_cell/bias/Adam/Assign5bidirectional_rnn/bw/basic_lstm_cell/bias/Adam/read:02Bbidirectional_rnn/bw/basic_lstm_cell/bias/Adam/Initializer/zeros:0
ě
2bidirectional_rnn/bw/basic_lstm_cell/bias/Adam_1:07bidirectional_rnn/bw/basic_lstm_cell/bias/Adam_1/Assign7bidirectional_rnn/bw/basic_lstm_cell/bias/Adam_1/read:02Dbidirectional_rnn/bw/basic_lstm_cell/bias/Adam_1/Initializer/zeros:0
\
Weights/Adam:0Weights/Adam/AssignWeights/Adam/read:02 Weights/Adam/Initializer/zeros:0
d
Weights/Adam_1:0Weights/Adam_1/AssignWeights/Adam_1/read:02"Weights/Adam_1/Initializer/zeros:0
P
Bias/Adam:0Bias/Adam/AssignBias/Adam/read:02Bias/Adam/Initializer/zeros:0
X
Bias/Adam_1:0Bias/Adam_1/AssignBias/Adam_1/read:02Bias/Adam_1/Initializer/zeros:0"˙Ŕ
while_contextěŔčŔ
Š`
+bidirectional_rnn/fw/fw/while/while_context *(bidirectional_rnn/fw/fw/while/LoopCond:02%bidirectional_rnn/fw/fw/while/Merge:0:(bidirectional_rnn/fw/fw/while/Identity:0B$bidirectional_rnn/fw/fw/while/Exit:0B&bidirectional_rnn/fw/fw/while/Exit_1:0B&bidirectional_rnn/fw/fw/while/Exit_2:0B&bidirectional_rnn/fw/fw/while/Exit_3:0B&bidirectional_rnn/fw/fw/while/Exit_4:0Bgradients/f_count_2:0J[
0bidirectional_rnn/fw/basic_lstm_cell/bias/read:0
2bidirectional_rnn/fw/basic_lstm_cell/kernel/read:0
%bidirectional_rnn/fw/fw/CheckSeqLen:0
!bidirectional_rnn/fw/fw/Minimum:0
%bidirectional_rnn/fw/fw/TensorArray:0
Tbidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
'bidirectional_rnn/fw/fw/TensorArray_1:0
'bidirectional_rnn/fw/fw/strided_slice:0
%bidirectional_rnn/fw/fw/while/Enter:0
'bidirectional_rnn/fw/fw/while/Enter_1:0
'bidirectional_rnn/fw/fw/while/Enter_2:0
'bidirectional_rnn/fw/fw/while/Enter_3:0
'bidirectional_rnn/fw/fw/while/Enter_4:0
$bidirectional_rnn/fw/fw/while/Exit:0
&bidirectional_rnn/fw/fw/while/Exit_1:0
&bidirectional_rnn/fw/fw/while/Exit_2:0
&bidirectional_rnn/fw/fw/while/Exit_3:0
&bidirectional_rnn/fw/fw/while/Exit_4:0
2bidirectional_rnn/fw/fw/while/GreaterEqual/Enter:0
,bidirectional_rnn/fw/fw/while/GreaterEqual:0
(bidirectional_rnn/fw/fw/while/Identity:0
*bidirectional_rnn/fw/fw/while/Identity_1:0
*bidirectional_rnn/fw/fw/while/Identity_2:0
*bidirectional_rnn/fw/fw/while/Identity_3:0
*bidirectional_rnn/fw/fw/while/Identity_4:0
*bidirectional_rnn/fw/fw/while/Less/Enter:0
$bidirectional_rnn/fw/fw/while/Less:0
,bidirectional_rnn/fw/fw/while/Less_1/Enter:0
&bidirectional_rnn/fw/fw/while/Less_1:0
*bidirectional_rnn/fw/fw/while/LogicalAnd:0
(bidirectional_rnn/fw/fw/while/LoopCond:0
%bidirectional_rnn/fw/fw/while/Merge:0
%bidirectional_rnn/fw/fw/while/Merge:1
'bidirectional_rnn/fw/fw/while/Merge_1:0
'bidirectional_rnn/fw/fw/while/Merge_1:1
'bidirectional_rnn/fw/fw/while/Merge_2:0
'bidirectional_rnn/fw/fw/while/Merge_2:1
'bidirectional_rnn/fw/fw/while/Merge_3:0
'bidirectional_rnn/fw/fw/while/Merge_3:1
'bidirectional_rnn/fw/fw/while/Merge_4:0
'bidirectional_rnn/fw/fw/while/Merge_4:1
-bidirectional_rnn/fw/fw/while/NextIteration:0
/bidirectional_rnn/fw/fw/while/NextIteration_1:0
/bidirectional_rnn/fw/fw/while/NextIteration_2:0
/bidirectional_rnn/fw/fw/while/NextIteration_3:0
/bidirectional_rnn/fw/fw/while/NextIteration_4:0
,bidirectional_rnn/fw/fw/while/Select/Enter:0
&bidirectional_rnn/fw/fw/while/Select:0
(bidirectional_rnn/fw/fw/while/Select_1:0
(bidirectional_rnn/fw/fw/while/Select_2:0
&bidirectional_rnn/fw/fw/while/Switch:0
&bidirectional_rnn/fw/fw/while/Switch:1
(bidirectional_rnn/fw/fw/while/Switch_1:0
(bidirectional_rnn/fw/fw/while/Switch_1:1
(bidirectional_rnn/fw/fw/while/Switch_2:0
(bidirectional_rnn/fw/fw/while/Switch_2:1
(bidirectional_rnn/fw/fw/while/Switch_3:0
(bidirectional_rnn/fw/fw/while/Switch_3:1
(bidirectional_rnn/fw/fw/while/Switch_4:0
(bidirectional_rnn/fw/fw/while/Switch_4:1
7bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter:0
9bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1:0
1bidirectional_rnn/fw/fw/while/TensorArrayReadV3:0
Ibidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
Cbidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3:0
%bidirectional_rnn/fw/fw/while/add/y:0
#bidirectional_rnn/fw/fw/while/add:0
'bidirectional_rnn/fw/fw/while/add_1/y:0
%bidirectional_rnn/fw/fw/while/add_1:0
3bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add:0
5bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_1:0
=bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd/Enter:0
7bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd:0
5bidirectional_rnn/fw/fw/while/basic_lstm_cell/Const:0
7bidirectional_rnn/fw/fw/while/basic_lstm_cell/Const_1:0
7bidirectional_rnn/fw/fw/while/basic_lstm_cell/Const_2:0
<bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul/Enter:0
6bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul:0
3bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul:0
5bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1:0
5bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2:0
7bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid:0
9bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid_1:0
9bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid_2:0
4bidirectional_rnn/fw/fw/while/basic_lstm_cell/Tanh:0
6bidirectional_rnn/fw/fw/while/basic_lstm_cell/Tanh_1:0
;bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat/axis:0
6bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat:0
5bidirectional_rnn/fw/fw/while/basic_lstm_cell/split:0
5bidirectional_rnn/fw/fw/while/basic_lstm_cell/split:1
5bidirectional_rnn/fw/fw/while/basic_lstm_cell/split:2
5bidirectional_rnn/fw/fw/while/basic_lstm_cell/split:3
,bidirectional_rnn/fw/fw/while/dropout/Cast:0
4bidirectional_rnn/fw/fw/while/dropout/GreaterEqual:0
-bidirectional_rnn/fw/fw/while/dropout/Shape:0
+bidirectional_rnn/fw/fw/while/dropout/mul:0
-bidirectional_rnn/fw/fw/while/dropout/mul_1:0
Dbidirectional_rnn/fw/fw/while/dropout/random_uniform/RandomUniform:0
:bidirectional_rnn/fw/fw/while/dropout/random_uniform/max:0
:bidirectional_rnn/fw/fw/while/dropout/random_uniform/min:0
:bidirectional_rnn/fw/fw/while/dropout/random_uniform/mul:0
:bidirectional_rnn/fw/fw/while/dropout/random_uniform/sub:0
6bidirectional_rnn/fw/fw/while/dropout/random_uniform:0
,bidirectional_rnn/fw/fw/while/dropout/rate:0
-bidirectional_rnn/fw/fw/while/dropout/sub/x:0
+bidirectional_rnn/fw/fw/while/dropout/sub:0
1bidirectional_rnn/fw/fw/while/dropout/truediv/x:0
/bidirectional_rnn/fw/fw/while/dropout/truediv:0
bidirectional_rnn/fw/fw/zeros:0
gradients/Add/y:0
gradients/Add:0
gradients/Merge:0
gradients/Merge:1
gradients/NextIteration:0
gradients/Switch:0
gradients/Switch:1
Dgradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/Enter:0
Jgradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPushV2:0
Dgradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/f_acc:0
jgradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter:0
pgradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2:0
jgradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc:0
Tgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/MatMul_1/Enter:0
Zgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2:0
Tgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc:0
Ngradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul/Enter:0
Tgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul/StackPushV2:0
Ngradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul/f_acc:0
Pgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul_1/Enter:0
Vgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPushV2:0
Pgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul_1/f_acc:0
Ngradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul/Enter:0
Tgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul/StackPushV2:0
Ngradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul/f_acc:0
Pgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul_1/Enter:0
Vgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPushV2:0
Pgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul_1/f_acc:0
Lgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul/Enter:0
Rgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul/StackPushV2:0
Lgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul/f_acc:0
Ngradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul_1/Enter:0
Tgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul_1/StackPushV2:0
Ngradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul_1/f_acc:0
Fgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul/Enter:0
Lgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul/StackPushV2:0
Fgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul/f_acc:0
Hgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul_1/Enter:0
Ngradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul_1/StackPushV2:0
Hgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul_1/f_acc:0
Dgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/Enter:0
Jgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/StackPushV2:0
Dgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/f_acc:0
Fgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/Enter:0
Lgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/StackPushV2:0
Fgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/f_acc:0
gradients/f_count:0
gradients/f_count_1:0
gradients/f_count_2:0Ź
Tgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc:0Tgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul_grad/MatMul_1/Enter:0Q
!bidirectional_rnn/fw/fw/Minimum:0,bidirectional_rnn/fw/fw/while/Less_1/Enter:0 
Ngradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul/f_acc:0Ngradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul/Enter:0b
'bidirectional_rnn/fw/fw/TensorArray_1:07bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter:0¤
Pgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul_1/f_acc:0Pgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2_grad/Mul_1/Enter:0r
%bidirectional_rnn/fw/fw/TensorArray:0Ibidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
Tbidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:09bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1:0
Dgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/f_acc:0Dgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/Enter:0O
bidirectional_rnn/fw/fw/zeros:0,bidirectional_rnn/fw/fw/while/Select/Enter:0 
Ngradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul/f_acc:0Ngradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul/Enter:0
Dgradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/f_acc:0Dgradients/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/Enter:0¤
Pgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul_1/f_acc:0Pgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1_grad/Mul_1/Enter:0
Fgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul/f_acc:0Fgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul/Enter:0r
2bidirectional_rnn/fw/basic_lstm_cell/kernel/read:0<bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul/Enter:0
Fgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/f_acc:0Fgradients/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/Enter:0q
0bidirectional_rnn/fw/basic_lstm_cell/bias/read:0=bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd/Enter:0Ř
jgradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc:0jgradients/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter:0
Lgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul/f_acc:0Lgradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul/Enter:0U
'bidirectional_rnn/fw/fw/strided_slice:0*bidirectional_rnn/fw/fw/while/Less/Enter:0 
Ngradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul_1/f_acc:0Ngradients/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_grad/Mul_1/Enter:0[
%bidirectional_rnn/fw/fw/CheckSeqLen:02bidirectional_rnn/fw/fw/while/GreaterEqual/Enter:0
Hgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul_1/f_acc:0Hgradients/bidirectional_rnn/fw/fw/while/dropout/mul_1_grad/Mul_1/Enter:0R%bidirectional_rnn/fw/fw/while/Enter:0R'bidirectional_rnn/fw/fw/while/Enter_1:0R'bidirectional_rnn/fw/fw/while/Enter_2:0R'bidirectional_rnn/fw/fw/while/Enter_3:0R'bidirectional_rnn/fw/fw/while/Enter_4:0Rgradients/f_count_1:0Z'bidirectional_rnn/fw/fw/strided_slice:0
š`
+bidirectional_rnn/bw/bw/while/while_context *(bidirectional_rnn/bw/bw/while/LoopCond:02%bidirectional_rnn/bw/bw/while/Merge:0:(bidirectional_rnn/bw/bw/while/Identity:0B$bidirectional_rnn/bw/bw/while/Exit:0B&bidirectional_rnn/bw/bw/while/Exit_1:0B&bidirectional_rnn/bw/bw/while/Exit_2:0B&bidirectional_rnn/bw/bw/while/Exit_3:0B&bidirectional_rnn/bw/bw/while/Exit_4:0Bgradients/f_count_5:0J˘[
0bidirectional_rnn/bw/basic_lstm_cell/bias/read:0
2bidirectional_rnn/bw/basic_lstm_cell/kernel/read:0
%bidirectional_rnn/bw/bw/CheckSeqLen:0
!bidirectional_rnn/bw/bw/Minimum:0
%bidirectional_rnn/bw/bw/TensorArray:0
Tbidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
'bidirectional_rnn/bw/bw/TensorArray_1:0
'bidirectional_rnn/bw/bw/strided_slice:0
%bidirectional_rnn/bw/bw/while/Enter:0
'bidirectional_rnn/bw/bw/while/Enter_1:0
'bidirectional_rnn/bw/bw/while/Enter_2:0
'bidirectional_rnn/bw/bw/while/Enter_3:0
'bidirectional_rnn/bw/bw/while/Enter_4:0
$bidirectional_rnn/bw/bw/while/Exit:0
&bidirectional_rnn/bw/bw/while/Exit_1:0
&bidirectional_rnn/bw/bw/while/Exit_2:0
&bidirectional_rnn/bw/bw/while/Exit_3:0
&bidirectional_rnn/bw/bw/while/Exit_4:0
2bidirectional_rnn/bw/bw/while/GreaterEqual/Enter:0
,bidirectional_rnn/bw/bw/while/GreaterEqual:0
(bidirectional_rnn/bw/bw/while/Identity:0
*bidirectional_rnn/bw/bw/while/Identity_1:0
*bidirectional_rnn/bw/bw/while/Identity_2:0
*bidirectional_rnn/bw/bw/while/Identity_3:0
*bidirectional_rnn/bw/bw/while/Identity_4:0
*bidirectional_rnn/bw/bw/while/Less/Enter:0
$bidirectional_rnn/bw/bw/while/Less:0
,bidirectional_rnn/bw/bw/while/Less_1/Enter:0
&bidirectional_rnn/bw/bw/while/Less_1:0
*bidirectional_rnn/bw/bw/while/LogicalAnd:0
(bidirectional_rnn/bw/bw/while/LoopCond:0
%bidirectional_rnn/bw/bw/while/Merge:0
%bidirectional_rnn/bw/bw/while/Merge:1
'bidirectional_rnn/bw/bw/while/Merge_1:0
'bidirectional_rnn/bw/bw/while/Merge_1:1
'bidirectional_rnn/bw/bw/while/Merge_2:0
'bidirectional_rnn/bw/bw/while/Merge_2:1
'bidirectional_rnn/bw/bw/while/Merge_3:0
'bidirectional_rnn/bw/bw/while/Merge_3:1
'bidirectional_rnn/bw/bw/while/Merge_4:0
'bidirectional_rnn/bw/bw/while/Merge_4:1
-bidirectional_rnn/bw/bw/while/NextIteration:0
/bidirectional_rnn/bw/bw/while/NextIteration_1:0
/bidirectional_rnn/bw/bw/while/NextIteration_2:0
/bidirectional_rnn/bw/bw/while/NextIteration_3:0
/bidirectional_rnn/bw/bw/while/NextIteration_4:0
,bidirectional_rnn/bw/bw/while/Select/Enter:0
&bidirectional_rnn/bw/bw/while/Select:0
(bidirectional_rnn/bw/bw/while/Select_1:0
(bidirectional_rnn/bw/bw/while/Select_2:0
&bidirectional_rnn/bw/bw/while/Switch:0
&bidirectional_rnn/bw/bw/while/Switch:1
(bidirectional_rnn/bw/bw/while/Switch_1:0
(bidirectional_rnn/bw/bw/while/Switch_1:1
(bidirectional_rnn/bw/bw/while/Switch_2:0
(bidirectional_rnn/bw/bw/while/Switch_2:1
(bidirectional_rnn/bw/bw/while/Switch_3:0
(bidirectional_rnn/bw/bw/while/Switch_3:1
(bidirectional_rnn/bw/bw/while/Switch_4:0
(bidirectional_rnn/bw/bw/while/Switch_4:1
7bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter:0
9bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1:0
1bidirectional_rnn/bw/bw/while/TensorArrayReadV3:0
Ibidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
Cbidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3:0
%bidirectional_rnn/bw/bw/while/add/y:0
#bidirectional_rnn/bw/bw/while/add:0
'bidirectional_rnn/bw/bw/while/add_1/y:0
%bidirectional_rnn/bw/bw/while/add_1:0
3bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add:0
5bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_1:0
=bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd/Enter:0
7bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd:0
5bidirectional_rnn/bw/bw/while/basic_lstm_cell/Const:0
7bidirectional_rnn/bw/bw/while/basic_lstm_cell/Const_1:0
7bidirectional_rnn/bw/bw/while/basic_lstm_cell/Const_2:0
<bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul/Enter:0
6bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul:0
3bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul:0
5bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1:0
5bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2:0
7bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid:0
9bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid_1:0
9bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid_2:0
4bidirectional_rnn/bw/bw/while/basic_lstm_cell/Tanh:0
6bidirectional_rnn/bw/bw/while/basic_lstm_cell/Tanh_1:0
;bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat/axis:0
6bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat:0
5bidirectional_rnn/bw/bw/while/basic_lstm_cell/split:0
5bidirectional_rnn/bw/bw/while/basic_lstm_cell/split:1
5bidirectional_rnn/bw/bw/while/basic_lstm_cell/split:2
5bidirectional_rnn/bw/bw/while/basic_lstm_cell/split:3
,bidirectional_rnn/bw/bw/while/dropout/Cast:0
4bidirectional_rnn/bw/bw/while/dropout/GreaterEqual:0
-bidirectional_rnn/bw/bw/while/dropout/Shape:0
+bidirectional_rnn/bw/bw/while/dropout/mul:0
-bidirectional_rnn/bw/bw/while/dropout/mul_1:0
Dbidirectional_rnn/bw/bw/while/dropout/random_uniform/RandomUniform:0
:bidirectional_rnn/bw/bw/while/dropout/random_uniform/max:0
:bidirectional_rnn/bw/bw/while/dropout/random_uniform/min:0
:bidirectional_rnn/bw/bw/while/dropout/random_uniform/mul:0
:bidirectional_rnn/bw/bw/while/dropout/random_uniform/sub:0
6bidirectional_rnn/bw/bw/while/dropout/random_uniform:0
,bidirectional_rnn/bw/bw/while/dropout/rate:0
-bidirectional_rnn/bw/bw/while/dropout/sub/x:0
+bidirectional_rnn/bw/bw/while/dropout/sub:0
1bidirectional_rnn/bw/bw/while/dropout/truediv/x:0
/bidirectional_rnn/bw/bw/while/dropout/truediv:0
bidirectional_rnn/bw/bw/zeros:0
gradients/Add_1/y:0
gradients/Add_1:0
gradients/Merge_2:0
gradients/Merge_2:1
gradients/NextIteration_2:0
gradients/Switch_2:0
gradients/Switch_2:1
Dgradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/Enter:0
Jgradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPushV2:0
Dgradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/f_acc:0
jgradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter:0
pgradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2:0
jgradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc:0
Tgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/MatMul_1/Enter:0
Zgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2:0
Tgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc:0
Ngradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul/Enter:0
Tgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul/StackPushV2:0
Ngradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul/f_acc:0
Pgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul_1/Enter:0
Vgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPushV2:0
Pgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul_1/f_acc:0
Ngradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul/Enter:0
Tgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul/StackPushV2:0
Ngradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul/f_acc:0
Pgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul_1/Enter:0
Vgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPushV2:0
Pgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul_1/f_acc:0
Lgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul/Enter:0
Rgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul/StackPushV2:0
Lgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul/f_acc:0
Ngradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul_1/Enter:0
Tgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul_1/StackPushV2:0
Ngradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul_1/f_acc:0
Fgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul/Enter:0
Lgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul/StackPushV2:0
Fgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul/f_acc:0
Hgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul_1/Enter:0
Ngradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul_1/StackPushV2:0
Hgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul_1/f_acc:0
Dgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/Enter:0
Jgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/StackPushV2:0
Dgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/f_acc:0
Fgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/Enter:0
Lgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/StackPushV2:0
Fgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/f_acc:0
gradients/f_count_3:0
gradients/f_count_4:0
gradients/f_count_5:0O
bidirectional_rnn/bw/bw/zeros:0,bidirectional_rnn/bw/bw/while/Select/Enter:0¤
Pgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul_1/f_acc:0Pgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul_1/Enter:0 
Ngradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul_1/f_acc:0Ngradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul_1/Enter:0
Fgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/f_acc:0Fgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/Enter:0Ř
jgradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc:0jgradients/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter:0[
%bidirectional_rnn/bw/bw/CheckSeqLen:02bidirectional_rnn/bw/bw/while/GreaterEqual/Enter:0Q
!bidirectional_rnn/bw/bw/Minimum:0,bidirectional_rnn/bw/bw/while/Less_1/Enter:0U
'bidirectional_rnn/bw/bw/strided_slice:0*bidirectional_rnn/bw/bw/while/Less/Enter:0 
Ngradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul/f_acc:0Ngradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1_grad/Mul/Enter:0r
%bidirectional_rnn/bw/bw/TensorArray:0Ibidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
Lgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul/f_acc:0Lgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_grad/Mul/Enter:0
Dgradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/f_acc:0Dgradients/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/Enter:0q
0bidirectional_rnn/bw/basic_lstm_cell/bias/read:0=bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd/Enter:0 
Ngradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul/f_acc:0Ngradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul/Enter:0
Dgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/f_acc:0Dgradients/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/Enter:0r
2bidirectional_rnn/bw/basic_lstm_cell/kernel/read:0<bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul/Enter:0
Tbidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:09bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1:0¤
Pgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul_1/f_acc:0Pgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2_grad/Mul_1/Enter:0
Hgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul_1/f_acc:0Hgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul_1/Enter:0
Fgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul/f_acc:0Fgradients/bidirectional_rnn/bw/bw/while/dropout/mul_1_grad/Mul/Enter:0b
'bidirectional_rnn/bw/bw/TensorArray_1:07bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter:0Ź
Tgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc:0Tgradients/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul_grad/MatMul_1/Enter:0R%bidirectional_rnn/bw/bw/while/Enter:0R'bidirectional_rnn/bw/bw/while/Enter_1:0R'bidirectional_rnn/bw/bw/while/Enter_2:0R'bidirectional_rnn/bw/bw/while/Enter_3:0R'bidirectional_rnn/bw/bw/while/Enter_4:0Rgradients/f_count_4:0Z'bidirectional_rnn/bw/bw/strided_slice:0"	
trainable_variablesôń
[
Embedding:0Embedding/AssignEmbedding/read:02&Embedding/Initializer/random_uniform:08
ă
-bidirectional_rnn/fw/basic_lstm_cell/kernel:02bidirectional_rnn/fw/basic_lstm_cell/kernel/Assign2bidirectional_rnn/fw/basic_lstm_cell/kernel/read:02Hbidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform:08
Ň
+bidirectional_rnn/fw/basic_lstm_cell/bias:00bidirectional_rnn/fw/basic_lstm_cell/bias/Assign0bidirectional_rnn/fw/basic_lstm_cell/bias/read:02=bidirectional_rnn/fw/basic_lstm_cell/bias/Initializer/zeros:08
ă
-bidirectional_rnn/bw/basic_lstm_cell/kernel:02bidirectional_rnn/bw/basic_lstm_cell/kernel/Assign2bidirectional_rnn/bw/basic_lstm_cell/kernel/read:02Hbidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform:08
Ň
+bidirectional_rnn/bw/basic_lstm_cell/bias:00bidirectional_rnn/bw/basic_lstm_cell/bias/Assign0bidirectional_rnn/bw/basic_lstm_cell/bias/read:02=bidirectional_rnn/bw/basic_lstm_cell/bias/Initializer/zeros:08
S
	Weights:0Weights/AssignWeights/read:02$Weights/Initializer/random_uniform:08
G
Bias:0Bias/AssignBias/read:02!Bias/Initializer/random_uniform:08"
train_op

Adam;AŹ