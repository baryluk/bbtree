-module(bbtree).
-author('baryluk@smp.if.uj.edu.pl').

-ifdef(TEST).
-include_lib("eunit/include/eunit.hrl").
-endif.

-compile({no_auto_import,[size/1]}).

% Implementation of bounded balance binary search trees
%
% @inproceedings{804906,
% author = {Nievergelt, J. and Reingold, E. M.},
% title = {Binary search trees of bounded balance},
% booktitle = {STOC '72: Proceedings of the fourth annual ACM symposium on Theory of computing},
% year = {1972},
% pages = {137--142},
% location = {Denver, Colorado, United States},
% doi = {http://doi.acm.org/10.1145/800152.804906},
% publisher = {ACM},
% address = {New York, NY, USA},
% }
%
%
% based on "Implementing Sets Efficiently in a Functional Language", Stephen Adams, CSTR 92-10
% http://groups.csail.mit.edu/mac/users/adams/BB/
%
% We belive this tree is good when there is big number of insertions,
% as insertion are pretty fast, due to the fact that balancing occurs
% not very frequently.
% Deletion performs balancing similary as insertion.
% If one wants to deletion not to perform balancing, use remove_no_balancing/2.
% If some one knows that after many insertions or deletions,
% subsequently there will be many many lookups, one can improve balancing,
% by improve_balance/1, which will try to make it slightly more balanced than
% just when inserting. It should only be used if number of lookup
% after it will be about 10 times the number of elements in the tree.
% It is unacassary to do it, if only fold will be used, as fold works same fast
% regradles of unbalance (stack space usage will eventually grow).
%
% This module can be used as a drop-in replacement for dict or gb_trees, as it provides both apis.
% It also provides new functions.
%
% It will also behave very good on random keys insertion, as random keys insertion
% produces balanced trees by itself.
%
% As such it can be used both as key-value lookup table, sets, multisets.
%
% Suppor for range queries will be enabled in the future versions.

-export([
	empty/0,             % empty() -> BB.                                                      % from gb_trees
	new/0,               % same as empty/0                                                     % from dict
	is_bb/1,             % is_bb(BB) -> true | false.
	is_empty/1,          % is_empty(BB) -> true | false.                                       % from dict and gb_trees
	size/1,              % size(BB) -> integer().                                              % from dict and gb_trees
	is_key/2,            % is_key(Key, BB) -> true | false.                                    % from dict
	is_defined/2,        % same as is_key/2                                                    % from gb_trees and proplists
	is_member/2,         % same as is_key/2                                                    % from gb_sets
	is_element/2,        % same as is_key/2                                                    % from gb_sets
	store/3,             % store(Key, Value, BB) -> BB2.                                       % from dict
	enter/3,             % same as store/3                                                     % from gb_trees
	insert/3,            % insert(Key, Value, BB) -> BB2 | crash.                              % from gb_trees
%	save/3,              % save(Key, Value, BB) -> {none | {ok, OldValue}, BB2}.
	erase/2,             % erase(Key, BB) -> BB2 | crash.                                      % from dict
	delete/2,            % same as erase/2                                                     % from gb_trees
	delete_any/2,        % delete(Key, BB) -> BB2.                                             % from gb_trees
%	take/2,              % take(Key, BB) -> {OldValue, BB2} | crash.
%	take/3,              % take(Key, BB [, Default]) -> {OldValue | Default, BB2}.
	largest/1,           % largest(BB) -> {K, V} | crash.                                      % from gb_trees
	smallest/1,          %                                                                     % from gb_trees
	take_largest/1,      % take_largest(BB) -> {K, V, BB2} | crash.                            % from_gbrees
	take_smallest/1,     %                                                                     % from_gbrees
	fold/3,              % fold(fun(K,V,Acc1) -> Acc2 end, Acc0, BB) -> Acc3.                  % from dict
	foldl/3, foldr/3,    % foldl(fun(K,V,Acc1) -> Acc2 end, Acc0, BB) -> Acc3.
	foldl2/3, foldr2/3,  % foldl2(fun({K,V},Acc1) -> Acc2 end, Acc0, BB) -> Acc3.
	foldl3/3, foldr3/3,  % foldl3(fun(K,V,Acc1) -> {next, Acc2} | {cancel, Acc3} end, Acc0, BB) -> Acc3.
	double_foldll2/4,    % double_foldll2(fun(KV1, KV2, Acc1) -> Acc2 end, Acc0, BB1, BB2) -> Acc3.
	update/3,            % update(Key, fun(OldV) -> NewV end, BB) -> BB2 | crash.              % from dict
	update/4,            % update(Key, fun(OldV) -> NewV end, InitV, BB) -> BB2.               % from dict
	update_lazy/4,       % update_lazy(Key, fun(OldV) -> NewV end, fun() -> InitV end, BB) -> BB2.
	update_counter/3,    % update_counter(Key, Inc, BB) -> BB2.                                % from dict
	update_counter/4,    % update_counter(Key, Inc, Init, BB) -> BB2.
	to_list/1,           % to_list(BB) -> [{K1,V1},{K2,V2},...].                               % from dict and gb_trees
	fetch/2,             % fetch(Key, Dict) -> V | crash.                                      % from dict
	fetch/3,             % fetch(Key, Dict, Default) -> V | Default.
	fetch_keys/1,        % fetch_keys(BB) -> [K1,K2,...].                                      % from dict
	keys/1,              % keys(BB) -> [K1,K2,...].                                            % from gb_trees
	keys_r/1,            % same as keys/1, but list is reversed
	get_keys/1,          % same as keys/1                                                      % from proplists
	find/2,              % find(Key, BB) -> {ok, V} | error.                                   % from dict
	find/3,              % find(Key, BB, Default) -> {ok, V} | {default, Default}.
	get_value/2,         % get_value(Key, BB) -> V | undefined.                                % from proplists
	get_value/3,         % get_value(Key, BB, Default) -> V | Default.                         % from proplists
	lookup/2,            % lookup(Key, BB) -> {value, V} | none.                               % from gb_trees
	                     %  in proplists there is similary lookup(K,List) -> tuple() | none.
	get/2,               % get(Key, BB) -> Value | crash.                                      % from gb_trees
	from_orddict/1,      % from_orddict([{K1,V1},...]) -> BB.                                  % from gb_trees
	from_list/1,         % from_list([{K1,V1},...]) -> BB.                                     % from dict
	foreachl/2,          % foreachl(fun(K,V) -> xx end, BB) -> void().
	foreachr/2,          % foreachr(fun(K,V) -> xx end, BB) -> void().
	foreach/2,           % same as foreachl/2, similar to lists:foreach
	map/2,               % map(fun(K, V1) -> V2 end, BB) -> BB2.                               % from dict and gb_trees
	map_foldl/3,         % map_foldl(fun(K, V1, Acc) -> {V2, Acc2} end, Acc0, BB) -> {BB2, Acc3}.
	map_foldr/3,         % map_foldr(fun(K, V1, Acc) -> {V2, Acc2} end, Acc0, BB) -> {BB2, Acc3}.
	map_reduce/4,        % map_reduce(fun(K, V1) -> [X] end, fun(X1, X2) -> X3 end, X0, BB) -> X4.
	values/1,            % values(BB) -> [V1,V2,...]                                           % from gb_trees
	values_r/1,          % same as values/1, but list is reversed
	iterator/1,          % iterator(BB) -> iterator().                                         % from gb_trees
	iterator_reverse/1,  % iterator_reverse(BB) -> iterator().
	iterator_range_i/3,  % iterator_range_i(I, J, BB) -> iterator().
	next/1,              % next(Iterator) -> {K, V, Iterator2} | none.                         % from gb_trees
	singleton/2,         % singleton(Key, Value) -> BB.    % based on gb_sets:singleton/1
	difference/2,        % difference(BB1, BB2) -> BB3
	union/1,             % union([BB1, ...]) -> BB2.        % values from BB1 have priority    % based on gb_sets:union/1
	union/2,             % union(BB1, BB2) -> BB3.          % values from BB1 have priority
	union_simple/2,      % same as union/2, just simpler algorithm.
	union_slow/2,        % same as union/2, just simpler algorithm.
%	union/3,             % union(fun(Key, V1, V2) -> V3 end, BB1, BB2) -> BB3. % custom priotity
	intersection/1,      % intersection([BB1, ...]) -> BB2. % values from BB1 have priority    % based on gb_sets:intersection/1
	intersection/2,      % intersection(BB1, BB2) -> BB3.   % values from BB1 have priority
%	intersection/3,      % intersection(fun(Key, V1, V2) -> V3 end, BB1, BB2) -> BB3.
%	merge/3,             % same as intersection/3                                              % from dict
	is_equal/2,
	is_keys_equal/2,
	is_keys_subset/2,
%	filter/2,            % filter(fun(Key, Value) -> true | false end, BB) -> BB2.             % from dict
	index_k/2,           % index_k(N, BB) -> Key | crash.
	index_v/2,           % index_v(N, BB) -> Value | crash.
	index_kv/2,          % index_kv(N, BB) -> {Key, Value} | crash.
	rank_i/2,            % rank_i(Key, BB) -> N | crash.
	rank_iv/2,           % rank_iv(Key, BB) -> {N, Value} | crash.
%	subtree_i/3,         % subtree_i(I, J, BB) -> BB2 | crash.
%	subtree_k/3          % subtree_k(Key1, Key2, BB) -> BB2 | crash.
	subtree_k_size/3,    % subtree_k_size(Key1, Key2, BB) -> integer() | crash.
%	beetwen_i/3          % beetwen_i(I, J, BB) -> BB2.
%	beetwen_k/3          % beetwen_k(Key1, Key2, BB) -> BB2.
%	beetwen_k_size/3,    % beetwen_k_size(Key1, Key2, BB) -> integer().
%	lt_kv/2,             % lt_kv(Key1, BB) -> {Key, Value} | none.  % Key1 do not need to be in BB
%	gt_kv/2,             % gt_kv(Key1, BB) -> {Key, Value} | none.  % Key1 do not need to be in BB
%	le_kv/2,             % le_kv(Key1, BB) -> {Key, Value} | none.  % Key1 do not need to be in BB
%	ge_kv/2,             % ge_kv(Key1, BB) -> {Key, Value} | none.  % Key1 do not need to be in BB
	wrap/1,              % creates parametrized module
%	balance/0,           % fully rebalance with omega=5. default omega=10
%	balance/1,           % balance(omega) when omega >= 4.7


	table/1,             % for QLC
%	table_reverse/1,     % for QLC

% mainainer functions
	is_correct/1,
	stat/1
	]).

% recomenened api:
%
%   empty/0,
%   is_defined/2,
%   save/3,
%   
%

% Bounded Balance trees, are trees which tries to maintain
% balance up to the factor omega in the size of both subtrees.

% Do not change below 4.646, as smaller values needs some more sophisticated algorithms
% to determine how to perform balancing.
%-define(omega, 5).
-define(omega, 10).


% We do not use records for internal nodes, as it will increase memory usage due to the fact
% that tuple will have one element more.
%-record(bbnode, {size=1,kv=throw(internal_error),left=nil,right=nil}).

% This is both used for box/unbox beetwen external and internal format
% and also as M:F() form, as bbtree_wrap is name of a module, which implements parametrized module.
% This way we can use all returned trees from this modules, as parametrized modules:
%    A = bbtree:empty().
%    B = A:store(witek,b).
%    B = store(witek,b,A).
-record(bbtree_wrap, {tree=nil}).
-define(WRAP(X), #bbtree_wrap{tree=X}).

% same
%-define(WRAP(X), {bbtree_wrap,X}).


%-define(FULL1(KV,L,R), {KV,_,L,R}).
%-define(FULL(K,V,L,R), {{K,V},_,L,R}).
-define(MATCH_FULL1(KV,L,R), {KV,_,L,R}).
-define(MATCH_FULL(K,V,L,R), {{K,V},_,L,R}).

-define(SPARSE1(KV), KV).
-define(MATCH_SPARSE1(KV), KV = {_,_}).
-define(SPARSE(K,V), {K,V}).
-define(MATCH_SPARSE(K,V), {K,V}).

-define(FULLS1(KV,Size,L,R), {KV,Size,L,R}).
-define(MATCH_FULLS1(KV,Size,L,R), {KV,Size,L,R}).
-define(FULLS(K,V,Size,L,R), {{K,V},Size,L,R}).
-define(MATCH_FULLS(K,V,Size,L,R), {{K,V},Size,L,R}).

% Time complexity: O(1)
empty() ->
	?WRAP(nil).

% Time complexity: O(1)
new() ->
	empty().


singleton(Key, Value) ->
	Tree = n_({Key, Value}, nil, nil),
	?WRAP(Tree).


% Time complexity: O(1)
-compile({inline, size_/1}).
size_({_KV, Size, _Left, _Right}) ->
	Size;
size_(?MATCH_SPARSE1(_KV)) -> % sparse leaf
	1;
size_(nil) ->
	0.

% Time complexity: O(1)
size(?WRAP(Tree)) ->
	size_(Tree).


% Time complexity: O(1)
is_bb(?WRAP(_)) ->
	true;
is_bb(_) ->
	false.

% Time complexity: O(1)
is_empty(?WRAP(nil)) ->
	true;
is_empty(?WRAP(_)) ->
	false.

% Time complexity: O(log n)
is_key(Key, Tree) ->
	case find(Key, Tree) of
		{ok, _} -> true;
		none -> false
	end.

% Time complexity: O(log n)
is_defined(Key, Tree) ->
	is_key(Key, Tree).

% Time complexity: O(log n)
is_element(Key, Tree) ->
	is_key(Key, Tree).

% Time complexity: O(log n)
is_member(Key, Tree) ->
	is_key(Key, Tree).


find_(Key, _T = ?MATCH_FULL(K, Value, L, R)) ->
	if
		Key < K -> % go left
			find_(Key, L);
		K < Key -> % go right
			find_(Key, R);
		true ->
			{ok, Value}
	end;
find_(Key, _T = ?MATCH_SPARSE(K, Value)) ->
	if
		Key < K -> none;
		K < Key -> none;
		true -> {ok, Value}
	end;
find_(_Key, nil) ->
	none.


% Time complexity: O(log n)
find(Key, ?WRAP(Tree)) ->
	find_(Key, Tree).

% Time complexity: O(log n)
find(Key, ?WRAP(Tree), Default) ->
	case find_(Key, Tree) of
		R = {ok, _Value} -> R;
		none -> {ok, Default}
	end.

% Time complexity: O(log n)
fetch(Key, ?WRAP(Tree)) ->
	case find_(Key, Tree) of
		{ok, Value} -> Value;
		none -> throw(badarg)
	end.

% Time complexity: O(log n)
fetch(Key, ?WRAP(Tree), Default) ->
	case find_(Key, Tree) of
		{ok, Value} -> Value;
		none -> Default
	end.

% Time complexity: O(log n)
lookup(Key, ?WRAP(Tree)) ->
	case find_(Key, Tree) of
		{ok, Value} -> {value, Value};
		R = none -> R
	end.

% Time complexity: O(log n)
get(Key, ?WRAP(Tree)) ->
	case find_(Key, Tree) of
		{ok, Value} -> Value;
		none -> throw(function_clause) % same as in gb_trees
	end.

% Time complexity: O(log n)
get_value(Key, Tree) ->
	get_value(Key, Tree, undefined).

get_value(Key, ?WRAP(Tree), Default) ->
% Time complexity: O(log n)
	case find_(Key, Tree) of
		{ok, Value} -> Value;
		none -> Default
	end.

%%%% SIMPLE ITERATING


%% FOLDS 1 2

% Time complexity: O(n)
foldl_inorder_(Acc1, Fun, ?MATCH_FULL(K,V, L, R)) ->
	Acc2 = foldl_inorder_(Acc1, Fun, L),
	Acc3 = Fun(K, V, Acc2),
	Acc4 = foldl_inorder_(Acc3, Fun, R),
	Acc4;
foldl_inorder_(Acc1, Fun, ?MATCH_SPARSE(K,V)) ->
	Fun(K, V, Acc1);
foldl_inorder_(Acc, _Fun, nil) ->
	Acc.

% Time complexity: O(n)
foldr_inorder_(Acc1, Fun, ?MATCH_FULL(K,V, L, R)) ->
	Acc2 = foldr_inorder_(Acc1, Fun, R),
	Acc3 = Fun(K, V, Acc2),
	Acc4 = foldr_inorder_(Acc3, Fun, L),
	Acc4;
foldr_inorder_(Acc1, Fun, ?MATCH_SPARSE(K,V)) ->
	Fun(K, V, Acc1);
foldr_inorder_(Acc, _Fun, nil) ->
	Acc.

% Time complexity: O(n)
foldl2_inorder_(Acc1, Fun, ?MATCH_FULL1(KV, L, R)) ->
	Acc2 = foldl2_inorder_(Acc1, Fun, L),
	Acc3 = Fun(KV, Acc2),
	Acc4 = foldl2_inorder_(Acc3, Fun, R),
	Acc4;
foldl2_inorder_(Acc1, Fun, ?MATCH_SPARSE1(KV)) ->
	Fun(KV, Acc1);
foldl2_inorder_(Acc, _Fun, nil) ->
	Acc.

% Time complexity: O(n)
foldr2_inorder_(Acc1, Fun, ?MATCH_FULL1(KV, L, R)) ->
	Acc2 = foldr2_inorder_(Acc1, Fun, L),
	Acc3 = Fun(KV, Acc2),
	Acc4 = foldr2_inorder_(Acc3, Fun, R),
	Acc4;
foldr2_inorder_(Acc1, Fun, ?MATCH_SPARSE1(KV)) ->
	Fun(KV, Acc1);
foldr2_inorder_(Acc, _Fun, nil) ->
	Acc.


% Time complexity: O(n)
foldl(Fun, Acc0, ?WRAP(Tree)) when is_function(Fun, 3) ->
	foldl_inorder_(Acc0, Fun, Tree);
foldl(_,_,_) ->
	throw(badarg).

% Time complexity: O(n)
foldr(Fun, Acc0, ?WRAP(Tree)) when is_function(Fun, 3) ->
	foldr_inorder_(Acc0, Fun, Tree);
foldr(_,_,_) ->
	throw(badarg).

% Time complexity: O(n)
foldl2(Fun, Acc0, ?WRAP(Tree)) when is_function(Fun, 2) ->
	foldl2_inorder_(Acc0, Fun, Tree);
foldl2(_,_,_) ->
	throw(badarg).

% Time complexity: O(n)
foldr2(Fun, Acc0, ?WRAP(Tree)) when is_function(Fun, 2) ->
	foldr2_inorder_(Acc0, Fun, Tree);
foldr2(_,_,_) ->
	throw(badarg).


% Time complexity: O(n)
fold(Fun, Acc0, Tree) ->
	foldl(Fun, Acc0, Tree).

%% FOREACH

% Time complexity: O(n)
foreachl_inorder_(Fun, ?MATCH_FULL(K,V, L, R)) ->
	foreachl_inorder_(Fun, L),
	Fun(K, V),
	foreachl_inorder_(Fun, R);
foreachl_inorder_(Fun, ?MATCH_SPARSE(K,V)) ->
	Fun(K, V);
foreachl_inorder_(_Fun, nil) ->
	void.

% Time complexity: O(n)
foreachr_inorder_(Fun, ?MATCH_FULL(K,V, L, R)) ->
	foreachr_inorder_(Fun, R),
	Fun(K, V),
	foreachr_inorder_(Fun, L);
foreachr_inorder_(Fun, ?MATCH_SPARSE(K,V)) ->
	Fun(K, V);
foreachr_inorder_(_Fun, nil) ->
	void.


% Time complexity: O(n)
foreachl(Fun, ?WRAP(Tree)) when is_function(Fun, 2) ->
	foreachl_inorder_(Fun, Tree);
foreachl(_,_) ->
	throw(badarg).

% Time complexity: O(n)
foreachr(Fun, ?WRAP(Tree)) when is_function(Fun, 2) ->
	foreachr_inorder_(Fun, Tree);
foreachr(_,_) ->
	throw(badarg).

% Time complexity: O(n)
foreach(Fun, Tree) when is_function(Fun, 2) ->
	foreachl(Fun, Tree).

% CANCELABLE FOLDS

% Time complexity: O(n)
foldl3_inorder_(Acc1, Fun, ?MATCH_FULL(K,V, L, R)) ->
	case foldl3_inorder_(Acc1, Fun, L) of
		{next, Acc2} ->
			case Fun(K, V, Acc2) of
				{next, Acc3} ->
					foldl3_inorder_(Acc3, Fun, R);
				{cancel, _Acc3} = R ->
					R
			end;
		{cancel, _Acc2} = R ->
			R
	end;
foldl3_inorder_(Acc1, Fun, ?MATCH_SPARSE(K,V)) ->
	Fun(K, V, Acc1);
foldl3_inorder_(Acc, _Fun, nil) ->
	{next, Acc}.

% Time complexity: O(n)
foldr3_inorder_(Acc1, Fun, ?MATCH_FULL(K,V, L, R)) ->
	case foldr3_inorder_(Acc1, Fun, R) of
		{next, Acc2} ->
			case Fun(K, V, Acc2) of
				{next, Acc3} ->
					foldr3_inorder_(Acc3, Fun, L);
				{cancel, _Acc3} = R ->
					R
			end;
		{cancel, _Acc2} = R ->
			R
	end;
foldr3_inorder_(Acc1, Fun, ?MATCH_SPARSE(K,V)) ->
	Fun(K, V, Acc1);
foldr3_inorder_(Acc, _Fun, nil) ->
	{next, Acc}.


% Time complexity: O(n)
foldl3(Fun, Acc0, ?WRAP(Tree)) when is_function(Fun, 3) ->
	foldl3_inorder_(Acc0, Fun, Tree);
foldl3(_,_,_) ->
	throw(badarg).


% Time complexity: O(n)
foldr3(Fun, Acc0, ?WRAP(Tree)) when is_function(Fun, 3) ->
	foldr3_inorder_(Acc0, Fun, Tree);
foldr3(_,_,_) ->
	throw(badarg).

%%%% EXPORTING TO OTHER FORMAS

% Time complexity: O(n)
to_list(Tree) ->
	foldr2(fun(KV, Acc) -> [KV | Acc] end, [], Tree).

% Time complexity: O(n)
keys(Tree) ->
	foldr(fun(Key, _Value, Acc) -> [Key|Acc] end, [], Tree).

% Time complexity: O(n)
keys_r(Tree) ->
	foldl(fun(Key, _Value, Acc) -> [Key|Acc] end, [], Tree).

% Time complexity: O(n)
fetch_keys(Tree) ->
	keys(Tree).

% Time complexity: O(n)
get_keys(Tree) ->
	keys(Tree).

% Time complexity: O(n)
values(Tree) ->
	foldr(fun(_Key, Value, Acc) -> [Value|Acc] end, [], Tree).

% Time complexity: O(n)
values_r(Tree) ->
	foldl(fun(_Key, Value, Acc) -> [Value|Acc] end, [], Tree).


%%% MAPS  (Advanced iterations)

% Time complexity: O(n)
map_(Fun, ?MATCH_FULLS(K,V, Size, L, R)) ->
	NewL = map_(Fun, L),
	NewV = Fun(K, V),
	NewR = map_(Fun, R),
	?FULLS(K,NewV, Size, NewL, NewR);
map_(Fun, ?MATCH_SPARSE(K,V)) ->
	NewV = Fun(K, V),
	?SPARSE(K, NewV);
map_(_Fun, T = nil) ->
	T.


% Time complexity: O(n)
map_foldl_inorder_(Fun, ?MATCH_FULLS(K,V, Size, L, R), Acc1) ->
	{NewL, Acc2} = map_foldl_inorder_(Fun, L, Acc1),
	{NewV, Acc3} = Fun(K, V, Acc2),
	{NewR, Acc4} = map_foldl_inorder_(Fun, R, Acc3),
	NewTree = ?FULLS(K, NewV, Size, NewL, NewR),
	{NewTree, Acc4};
map_foldl_inorder_(Fun, ?MATCH_SPARSE(K,V), Acc1) ->
	{NewV, Acc2} = Fun(K, V, Acc1),
	{?SPARSE(K, NewV), Acc2};
map_foldl_inorder_(_Fun, T = nil, Acc) ->
	{T, Acc}.

% Time complexity: O(n)
map_foldr_inorder_(Fun, ?MATCH_FULLS(K,V, Size, L, R), Acc1) ->
	{NewR, Acc2} = map_foldr_inorder_(Fun, R, Acc1),
	{NewV, Acc3} = Fun(K, V, Acc2),
	{NewL, Acc4} = map_foldr_inorder_(Fun, L, Acc3),
	NewTree = ?FULLS(K, NewV, Size, NewL, NewR),
	{NewTree, Acc4};
map_foldr_inorder_(Fun, ?MATCH_SPARSE(K,V), Acc1) ->
	{NewV, Acc2} = Fun(K, V, Acc1),
	{?SPARSE(K, NewV), Acc2};
map_foldr_inorder_(_Fun, T = nil, Acc) ->
	{T, Acc}.



% Time complexity: O(n)
map(Fun, ?WRAP(Tree)) when is_function(Fun, 2) ->
	?WRAP(map_(Fun, Tree)).

% Time complexity: O(n)
map_foldl(Fun, Acc0, ?WRAP(Tree)) when is_function(Fun, 3) ->
	{T, A} = map_foldl_inorder_(Fun, Tree, Acc0),
	{?WRAP(T), A}.

% Time complexity: O(n)
map_foldr(Fun, Acc0, ?WRAP(Tree)) when is_function(Fun, 3) ->
	{T, A} = map_foldr_inorder_(Fun, Tree, Acc0),
	{?WRAP(T), A}.


% Time complexity: O(n)
map_reduce_(FunMap, FunReduce, Acc1, ?MATCH_FULL(K,V,L,R)) ->
	Acc2 = map_reduce(FunMap, FunReduce, Acc1, L),
	Mapped = FunMap(K, V),
	Acc3 = lists:foldl(FunReduce, Acc2, Mapped),
	Acc4 = map_reduce(FunMap, FunReduce, Acc3, R),
	Acc4;
map_reduce_(FunMap, FunReduce, Acc1, ?MATCH_SPARSE(K,V)) -> % sparse leaf
	Mapped = FunMap(K, V),
	Acc2 = lists:foldl(FunReduce, Acc1, Mapped),
	Acc2;
map_reduce_(_FunMap, _FunReduce, Acc, nil) ->
	Acc.

% Time complexity: O(n)
map_reduce(FunMap, FunReduce, Reduce0, ?WRAP(Tree)) when is_function(FunMap, 2), is_function(FunReduce, 2) ->
	map_reduce_(FunMap, FunReduce, Reduce0, Tree).

%%% INTERNALS

% simple constructor
% Time complexity: O(1)
-compile({inline, t_/4}).
t_(KV, N, L, R) ->
	?FULLS1(KV, N, L, R).

% smart constructor
% Time complexity: O(1)
-compile({inline, n_/3}).
n_(KV, nil, nil) -> % sparse leaf
	?SPARSE1(KV);
n_(KV, L, R) ->
	t_(KV, size_(L) + size_(R) + 1, L, R).


% rotation of nodes (used in t_prim)
% Time complexity: O(1)
-compile({inline, single_L_/1}).
single_L_(?MATCH_FULL1(KV_A, X, ?MATCH_FULL1(KV_B, Y, Z))) ->
	n_(KV_B, n_(KV_A, X, Y), Z).

-compile({inline, single_R_/1}).
single_R_(?MATCH_FULL1(KV_B, ?MATCH_FULL1(KV_A, X, Y), Z)) ->
	n_(KV_A, X, n_(KV_B, Y, Z)).

-compile({inline, double_L_/1}).
double_L_(?MATCH_FULL1(KV_A, X, ?MATCH_FULL1(KV_C, ?MATCH_FULL1(KV_B, Y1, Y2), Z))) ->
	n_(KV_B, n_(KV_A, X, Y1), n_(KV_C, Y2, Z)).

-compile({inline, double_R_/1}).
double_R_(?MATCH_FULL1(KV_C, ?MATCH_FULL1(KV_A, X, ?MATCH_FULL1(KV_B, Y1, Y2)), Z)) ->
	n_(KV_B, n_(KV_A, X, Y1), n_(KV_C, Y2, Z)).


% used when orginal tree was balances, and one of subtree changed by at most one element
% (by insertion or deletion of single element)
% Time complexity: O(1)
t_prim_(KV, L, R) ->
	P = n_(KV, L, R),

	LN = size_(L),
	RN = size_(R),
	if
		LN+RN < 2 ->
			P;
		RN > ?omega*LN ->   % right is too big
			{_, _, RL, RR} = R,
			RLN = size_(RL),
			RRN = size_(RR),
			if
				RLN < RRN ->
					single_L_(P);
				true ->
					double_L_(P)
			end;
		LN > ?omega*RN -> % left is too big
			{_, _, LL, LR} = L,
			LLN = size_(LL),
			LRN = size_(LR),
			if
				LRN < LLN ->
					single_R_(P);
				true ->
					double_R_(P)
			end;
		true ->
			P
	end.

% insertion
% Time complexity: O(log n)
add_(?MATCH_FULL1(KV_A = {K_A, _}, L, R), KV_X = {K_X, _}) ->
	if
		K_X < K_A -> % go left
			t_prim_(KV_A, add_(L, KV_X), R);
		K_A < K_X -> % go right
			t_prim_(KV_A, L, add_(R, KV_X));
		true ->
			n_(KV_X, L, R) % update
	end;
add_(?MATCH_SPARSE1(KV_A = {K_A, _}), KV_X = {K_X, _}) -> % sparse leaf
	L = nil,
	R = nil,
	if
		K_X < K_A -> % go left
			t_prim_(KV_A, add_(L, KV_X), R);
		K_A < K_X -> % go right
			t_prim_(KV_A, L, add_(R, KV_X));
		true ->
			n_(KV_X, L, R) % update
	end;
add_(nil, KV_X) ->
	n_(KV_X, nil, nil).

% external API for addintion
% Time complexity: O(log n)
store(Key, Value, ?WRAP(Tree)) ->
	?WRAP(add_(Tree, {Key, Value})).

% Time complexity: O(log n)
enter(Key, Value, Tree) ->
	store(Key, Value, Tree).

% Time complexity: O(log n)
insert(Key, Value, ?WRAP(Tree)) ->
	?WRAP(add_(Tree, {Key, Value})).

% deletion
% Time complexity: O(log n)
delete_(?MATCH_FULL1(KV_A = {K_A, _}, L, R), K_X) ->
	if
		K_X < K_A -> % go left
			t_prim_(KV_A, delete_(L, K_X), R);
		K_A < K_X -> % go right
			t_prim_(KV_A, L, delete_(R, K_X));
		true ->
			delete_prim_(L, R)
	end;
delete_(?MATCH_SPARSE1(KV_A = {K_A, _}), K_X) -> % sparse leaf
	L = nil,
	R = nil,
	if
		K_X < K_A -> % go left
			t_prim_(KV_A, delete_(L, K_X), R);
		K_A < K_X -> % go right
			t_prim_(KV_A, L, delete_(R, K_X));
		true ->
			delete_prim_(L, R)
	end;
delete_(T = nil, _K) ->
	T.

%delete_prim1_(nil, R) ->
%	R;
%delete_prim1_(L, nil) ->
%	L;
%delete_prim1_(L, R) ->
%	% we just find minimum element of R, remove it from R, and make it new root
%	KV_MinR = {K_MinR, _} = min_(R),
%	t_prim_(KV_MinR, L, delete_(R, K_MinR)).


% this is optimized version, in which we have specialized version for removing leftmost element
% Time complexity: O(log n)
%delete_prim_(nil, R) ->
%	R;
%delete_prim_(L, nil) ->
%	L;
%delete_prim_(L, R) ->
%	KV_MinR = min_(R),
%	% we just find minimum element of R, remove it from R, and make it new root
%	t_prim_(KV_MinR, L, delmin_balance_(R)).

% this is optimized more version, in which we have specialized version which removes leftmost element, and in the same time returns it
% Time complexity: O(log n)
delete_prim_(nil, R) ->
	R;
delete_prim_(L, nil) ->
	L;
delete_prim_(L, R) ->
	{KV_MinR, NewR} = delmin_take_balance_(R),
	% we just find minimum element of R, remove it from R, and make it new root
	t_prim_(KV_MinR, L, NewR).


% finds smallest element, return tree without it
% Time complexity: O(log n)
%delmin_balance_(?FULL1(_KV_Min,nil,R)) ->
%	R;
%delmin_balance_(?FULL1(KV,L,R)) ->
%	t_prim_(KV, delmin_balance_(L), R).

% finds largest element, return tree without it
% Time complexity: O(log n)
%delmax_balance_(?FULL1(_KV_Max,L,nil)) ->
%	L;
%delmax_balance_(?FULL1(KV,L,R)) ->
%	t_prim_(KV, L, delmax_balance_(R)).

% finds smallest element, return tuple: the smallest {K,V} and tree without it
% Time complexity: O(log n)
delmin_take_balance_(?MATCH_FULL1(KV_Min,nil,R)) ->
	{KV_Min, R};
delmin_take_balance_(?MATCH_FULL1(KV,L,R)) ->
	{KV_Min, NewL} = delmin_take_balance_(L),
	{KV_Min, t_prim_(KV, NewL, R)};
delmin_take_balance_(?MATCH_SPARSE1(KV_Min)) -> % sparse leaf
	R = nil,
	{KV_Min, R}.

% finds largest element, return tuple: the smallest {K,V} and tree without it
% Time complexity: O(log n)
delmax_take_balance_(?MATCH_FULL1(KV_Max,L,nil)) ->
	{KV_Max, L};
delmax_take_balance_(?MATCH_FULL1(KV,L,R)) ->
	{KV_Max, NewR} = delmax_take_balance_(R),
	{KV_Max, t_prim_(KV, L, NewR)};
delmax_take_balance_(?MATCH_SPARSE1(KV_Max)) -> % sparse leaf
	L = nil,
	{KV_Max, L}.



% external API for deletion
% Time complexity: O(log n)
erase(Key, ?WRAP(Tree)) ->
	?WRAP(delete_(Tree, Key)).

delete(Key, ?WRAP(Tree)) ->
	?WRAP(delete_(Tree, Key)).

delete_any(Key, ?WRAP(Tree)) ->
	?WRAP(delete_(Tree, Key)).


% Time complexity: O(log n)
min_(?MATCH_FULL1(KV, nil, _)) ->
	KV;
min_(?MATCH_FULL1(_KV, L, _)) ->
	min_(L);
min_(?MATCH_SPARSE1(KV)) ->
	KV.

% Time complexity: O(log n)
max_(?MATCH_FULL1(KV, _, nil)) ->
	KV;
max_(?MATCH_FULL1(_KV, _, R)) ->
	max_(R);
max_(?MATCH_SPARSE1(KV)) ->
	KV.

% Time complexity: O(log n)
smallest(?WRAP(Tree)) ->
	min_(Tree).

% Time complexity: O(log n)
largest(?WRAP(Tree)) ->
	max_(Tree).

% Time complexity: O(log n)
take_smallest(?WRAP(Tree)) ->
	{{K,V}, NewTree} = delmin_take_balance_(Tree),
	{K, V, ?WRAP(NewTree)}.

% Time complexity: O(log n)
take_largest(?WRAP(Tree)) ->
	{{K,V}, NewTree} = delmax_take_balance_(Tree),
	{K, V, ?WRAP(NewTree)}.



%%%% UPDATES

update_lazy(Key, Fun, InitFun, ?WRAP(Tree)) when is_function(Fun, 1), is_function(InitFun, 0) ->
	?WRAP(Tree).

% this 4 functions bellow should be specialized, so no unacassary funs are created.

update(Key, Fun, Tree) ->
	update_lazy(Key, Fun, fun() -> throw(badarg) end, Tree).

update(Key, Fun, Init, Tree) ->
	update_lazy(Key, Fun, fun() -> Init end, Tree).

update_counter(Key, Inc, Tree) when is_number(Inc) ->
	update_lazy(Key, fun(OldV) -> OldV+Inc end, fun() -> Inc end, Tree).

update_counter(Key, Inc, InitV, Tree) when is_number(Inc), is_number(InitV) ->
	update_lazy(Key, fun(OldV) -> OldV+Inc end, fun() -> InitV end, Tree).


%%%% IMPORT

from_list_([], Tree) ->
	Tree;
from_list_([H|T], Tree) ->
	from_list_(T, add_(Tree, H)).

from_list(List) ->
	?WRAP(from_list_(List, nil)).

from_orddict(List) ->
	from_list(List).

%%%% UNION



% Time complexity: O(n+m) - worst case
union_(nil, TreeB) -> TreeB;
union_(TreeA, _TreeB = ?MATCH_FULL1(KV = {K, _V}, L, R)) ->
	L2 = split_lt_(TreeA, K),
	R2 = split_gt_(TreeA, K),
	concat3_(KV, union_(L2, L), union_(R2, R));
union_(TreeA, nil) -> TreeA.

concat3_(KV, L = ?MATCH_FULLS1(KV_1, N_1, L_1, R_1), R = ?MATCH_FULLS1(KV_2, N_2, L_2, R_2)) ->
	if
		?omega*N_1 < N_2 ->
			t_prim_(KV_2, concat3_(KV, L, L_2), R_2);
		?omega*N_2 < N_1 ->
			t_prim_(KV_1, L_1, concat3_(KV, R_1, R));
		true ->
			n_(KV, L, R)
	end;
concat3_(KV, nil, R) ->
	add_(R, KV);
concat3_(KV, L, nil) ->
	add_(L, KV).



% returns tree with all those elements of Tree which are less than X
% Time complexity: O(log n)
split_lt_(?MATCH_FULL1(KV = {K, _V}, L, R), X) ->
	if
		X < K ->
			split_lt_(L, X);
		K < X ->
			concat3_(KV, L, split_lt_(R, X));
		true ->
			L
	end;
split_lt_(T = nil, _X) ->
	T.

% returns tree with all those elements of Tree which are greater than X
% Time complexity: O(log n)
split_gt_(?MATCH_FULL1(KV = {K, _V}, L, R), X) ->
	if
		K < X ->
			split_gt_(R, X);
		X < K ->
			concat3_(KV, split_gt_(L, X), R);
		true ->
			L
	end;
split_gt_(T = nil, _X) ->
	T.


% Time complexity: O(n+m) - worst case
union(?WRAP(TreeA), ?WRAP(TreeB)) ->
	?WRAP(union_prim_(TreeA, TreeB)).


union_slow(?WRAP(TreeA), ?WRAP(TreeB)) ->
	%?WRAP(union_(TreeA, TreeB, fun(_K, V_A, _V_B) -> V_A end)).
	?WRAP(union_(TreeA, TreeB)).


% Time complexity: O(n+m) - worst case
%union(?WRAP(TreeA), ?WRAP(TreeB), Fun) when is_function(Fun, 3) ->
%	?WRAP(union_(TreeA, TreeB, Fun)).

% Elegant and nice, but not very efficient.
% Time complexity: O(m (log n+m) )
union_simple(?WRAP(TreeA), ?WRAP(TreeB)) ->
	?WRAP(foldr2_inorder_(fun(KV, T) -> add_(T, KV) end, TreeA, TreeB)).


%%%% DIFFERENCE

difference_(TreeA = nil, _TreeB) ->
	TreeA;
difference_(TreeA, _TreeB = ?MATCH_FULL(K,_V, L, R)) ->
	L2 = split_lt_(TreeA, K),
	R2 = split_gt_(TreeA, K),
	concat_(difference_(L2, L), difference_(R2, R));
difference_(TreeA, nil) ->
	TreeA.

% simple concat
% Time complexity: O(log n)
%concat(TreeA, nil) ->
%	TreeA;
%concat(TreeA, TreeB) ->
%	{KV_MinB, NewTreeB} = delmin_take(TreeB),
%	concat3(KV_MinB, TreeA, NewTreeB).

% improved concat
% Time complexity: O(log n)
concat_(TreeA = ?MATCH_FULLS1(KV_1, N_1, L_1, R_1), TreeB = ?MATCH_FULLS1(KV_2, N_2, L_2, R_2)) ->
	if
		?omega*N_1 < N_2 ->
			t_prim_(KV_2, concat_(TreeA, L_2), R_2);
		?omega*N_2 < N_1 ->
			t_prim_(KV_1, L_1, concat_(R_1, TreeB));
		true ->
			{KV_MinB, NewTreeB} = delmin_take_balance_(TreeB),
			t_prim_(KV_MinB, TreeA, NewTreeB)
	end;
concat_(TreeA, nil) ->
	TreeA;
concat_(nil, TreeB) ->
	TreeB.


difference(?WRAP(TreeA), ?WRAP(TreeB)) ->
	?WRAP(difference_(TreeA, TreeB)).

%%%% INTERSECTION

% Returns common part of two trees.
%intersection(TreeA, TreeB) ->
%%	difference(TreeA, difference(TreeA, TreeB)).
%	difference(TreeB, difference(TreeB, TreeA)).

intersection_(TreeA = nil, _) ->
	TreeA;
intersection_(TreeA, _TreeB = ?MATCH_FULL1(KV = {K,V}, L, R)) ->
	L2 = split_lt_(TreeA, V),
	R2 = split_gt_(TreeA, V),
	case is_key(K, TreeA) of
		true ->
			concat3_(KV, intersection_(L2, L), intersection_(R2, R));
		false ->
			concat_(intersection_(L2, L), intersection_(R2, R))
	end;
intersection_(_, TreeB = nil) ->
	TreeB.


% improved intersection, single pass
intersection(?WRAP(TreeA), ?WRAP(TreeB)) ->
	?WRAP(intersection_(TreeA, TreeB)).

% hedge union. we are not constructing new subtrees (using split_lt, which uses concat3, which uses balancing t_prim)
% we are just using lazy subtree construction, by just noting Low,High elements to restrict tree
% we construct tree for real only at the bottom
% Time complexity: O(n log n) - worse than naive version!
union_prim_(_TreeA = ?MATCH_FULL1(KV = {K,_V}, L, R), {TreeB, Low, High}) ->
	concat3_(
		KV,
		union_prim_(L, {TreeB, Low, K}),
		union_prim_(R, {TreeB, K, High})
	);
union_prim_(nil, {TreeB, Low, High}) ->
	split_gt_lt_(TreeB, Low, High).

% returns tree with all those elements which are less than High and grater than High
-compile({inline, split_gt_lt_/3}).
split_gt_lt_(Tree, Low, High) ->
	split_gt_(split_lt_(Tree, High), Low).



intersection([H|T]) ->
	lists:foldl(fun(NextSet, Acc) ->
		intersection(Acc, NextSet)
	end, T, H).

union([H|T]) ->
	lists:foldl(fun(NextSet, Acc) ->
		union(Acc, NextSet)
	end, T, H).


is_equal(?WRAP(TreeA), ?WRAP(TreeB)) ->
	false.

is_keys_equal(?WRAP(TreeA), ?WRAP(TreeB)) ->
	false.

is_keys_subset(?WRAP(TreeA), ?WRAP(TreeB)) ->
	false.


% finds {K,V}, which is N'th in the Tree, if all K would be sorted
index_kv(N, ?WRAP(Tree)) when is_integer(N), N >= 1 ->
	{k,v}.

index_k(N, Tree) ->
	{K, _} = index_kv(N, Tree),
	K.

index_v(N, Tree) ->
	{_, V} = index_kv(N, Tree),
	V.


% tells what is the rank of K
rank_iv(K, ?WRAP(Tree)) ->
	{4, v}.

rank_i(K, ?WRAP(Tree)) ->
	4.


subtree_k_size(Key1, Key2, Tree) ->
	rank_i(Key2, Tree) - rank_i(Key1, Tree).


wrap(Tree) ->
	bbtree_wrap:new(Tree).

% create empty tree, which when used in later operations
% will use ordering by Fun, than by '<', '>' operators.
% Fun must be total ordering:
%    Fun(X,X) = false.
%    Fun(X,Y) and Fun(Y,X) = false.
%    Fun(A,B) and Fun(B,C) => Fun(A,C).
%
% Two keys are assumed to be equal if neighter is less than the other.
%
%empty_with_order(Fun) when is_function(Fun, 2) ->
%	{bbtree, Fun, nil}.




% It is similar to lists:zipwith/2
% Time complexity: O(n)
%double_foldll2_inorder_(Acc1, Fun, T1 = ?FULL1(KV1, L1, R1), T2 = ?FULL1(KV2, L2, R2)) ->
%	Acc2 = double_foldll2_inorder_(Acc1, Fun, L1, L2),
%	Acc3 = Fun(KV1, KV2, Acc2),
%	Acc4 = double_foldll2_inorder_(Acc3, Fun, R1, R2),
%	Acc4.
%double_foldll2_inorder_(Acc, _Fun, nil, nil) ->
%	Acc.
%

%double_foldl2(Fun, Acc0, ?WRAP(TreeA), ?WRAP(TreeB)) when is_function(Fun, 3) ->
%	double_foldl2_inorder_(Acc0, Fun, TreeA, TreeB).

% above should be equivalent to, but faster and without additional allocations
double_foldll2(Fun, Acc0, TreeA, TreeB) when is_function(Fun, 3) ->
	lists:foldl(
		fun({X,Y}, Acc) ->
			Fun(X, Y, Acc)
		end,
		Acc0,
		lists:zip(to_list(TreeA), to_list(TreeB))
	).

-ifdef(EUNIT).
double_foldl2_test() ->
	A1 = bbtree:from_list([{a,22},{b,33},{c,66},{d,77}]),
	?assert(is_correct(A1)),
	A2 = bbtree:from_list([{"basia",27},{"ala",13},{"witek",24},{"tomek",28}]),
	?assert(is_correct(A2)),
	A12 = bbtree:double_foldll2(
		fun({A,B}, {C,D}, Acc) ->
			[{A,C,B+D} | Acc]
		end,
		[],
		A1, A2),
	?assert(A12 =:= [{a,"ala",35},{b,"basia",60},{c,"tomek",94},{d,"witek",101}]),
	ok.
-endif.

% this checks if internal structure is correct
is_correct(?WRAP(Tree)) ->
	R1 = is_correct_bst_(Tree, 0, fun(_X) -> true end, root),
	{R2, _} = is_correct_size_(Tree, 0, root),
	{R3, _} = is_correct_bb_(Tree, 0),
	case {R1, R2, R3} of
		{true, true, true} -> true;
		Other -> Other
	end.

is_correct_bst_(nil, _Level, _Checker, _ParentDir) ->
	true;
is_correct_bst_(?MATCH_SPARSE(K,_V), _Level, Checker, _ParentDir) ->
	Checker(K);
is_correct_bst_(?MATCH_FULLS(K,_V, _S, L, R), Level, Checker, ParentDir) ->
	io:format("Testing1 at level ~p: K = ~p when ~p~n", [Level, K, ParentDir]),
	ResultK = Checker(K),
	ResultL = is_correct_bst_(L, Level+1, fun(X) ->
		case Checker(X) of
			true -> if X < K -> true; true -> {problem_L,K,X} end;
			Other -> Other
		end
	end, left),
	ResultR = is_correct_bst_(R, Level+1, fun(X) ->
		case Checker(X) of
			true -> if K < X -> true; true -> {problem_R,K,X} end;
			Other -> Other
		end
	end, right),
	case {ResultL, ResultK, ResultR} of
		{true, true, true} -> true;
		Other -> Other
	end.

is_correct_size_(nil, _Level, _ParentDir) ->
	{true, 0};
is_correct_size_(?MATCH_SPARSE(_K,_V), _Level, _ParentDir) ->
	{true, 1};
is_correct_size_(?MATCH_FULLS(_K,_V, S, L, R), Level, _ParentDir) ->
	{ResultL, SL} = is_correct_size_(L, Level+1, left),
	{ResultR, SR} = is_correct_size_(R, Level+1, right),
	io:format("Testing2 at level ~p:  SL=~p and SR=~p~n", [Level, SL, SR]),
	case {ResultL, ResultR, SL+SR+1} of
		{true, true, S} -> {true, S};
		Other -> Other
	end.

is_correct_bb_(nil, _Level) ->
	{true, 0};
is_correct_bb_(?MATCH_SPARSE(_K,_V), _Level) ->
	{true, 1};
is_correct_bb_(?MATCH_FULLS(_K,_V, S, L, R), Level) ->
	io:format("Testing3 at level ~p~n", [Level]),
	{_ResultL = true, SL} = is_correct_bb_(L, Level+1),
	{_ResultR = true, SR} = is_correct_bb_(R, Level+1),
	S = SL+SR+1,
	io:format("comparing3 at level ~p:  SL=~p and SR=~p~n", [Level, SL, SR]),
	if
		SL+SR < 2 -> {true, S};
		SL > ?omega*SR -> false;
		SR > ?omega*SL -> false;
		true -> {true, S}
	end.

stat(?WRAP(Tree)) ->
	stat_(Tree, {0,0,0,0,0}).

stat_(?MATCH_SPARSE(_,_), _Stat = {Full, Leaf, LeftOnly, RightOnly, SparseLeaf}) ->
	{Full, Leaf, LeftOnly, RightOnly, SparseLeaf+1};
stat_(?MATCH_FULL(_,_, nil, nil), _Stat = {Full, Leaf, LeftOnly, RightOnly, SparseLeaf}) ->
	{Full, Leaf+1, LeftOnly, RightOnly, SparseLeaf};
stat_(?MATCH_FULL(_,_, L, nil), _Stat = {Full, Leaf, LeftOnly, RightOnly, SparseLeaf}) ->
	stat_(L, {Full, Leaf, LeftOnly+1, RightOnly, SparseLeaf});
stat_(?MATCH_FULL(_,_, nil, R), _Stat = {Full, Leaf, LeftOnly, RightOnly, SparseLeaf}) ->
	stat_(R, {Full, Leaf, LeftOnly, RightOnly+1, SparseLeaf});
stat_(?MATCH_FULL(_,_, L, R), _Stat = {Full, Leaf, LeftOnly, RightOnly, SparseLeaf}) ->
	stat_(R, stat_(L, {Full+1, Leaf, LeftOnly, RightOnly, SparseLeaf})).


% Q3=bbtree:from_list([ {X,X} || X <- lists:seq(1,10000) ]).
% bbtree:stat(Q3).
% byte_size(term_to_binary(Q3)).


% for QLC
table(Tree, Opts) ->
	NN = case proplists:is_defined(reverse, Opts) of
		true -> next(iterator_reverse(Tree));
		false -> next(iterator(Tree))
	end,
	TraverseFun = fun() -> qlc_next(NN) end,
	InfoFun = fun
		(num_of_objects) -> size(Tree);
		(keypos) -> 1;
		(is_sorted_key) -> true;
		(is_unique_objects) -> true;
		(_) -> undefined
	end,
	FormatFun = fun
		({all, NElements, ElementFun}) ->
			ValsS = io_lib:format("bbtree:from_orddict(~w)", [qlc_nodes(Tree, NElements, ElementFun)]),
			io_lib:format("bbtree:table(~s)", [ValsS]);
		({lookup, 1, KeyValues, _NElements, ElementFun}) ->
			ValsS = io_lib:format("bbtree:from_orddict(~w)", [qlc_nodes(Tree, infinity, ElementFun)]),
			io_lib:format(
				"lists:flatmap(fun(K) -> "
				"case bbtree:lookup(K, ~s) of "
				"{value, V} -> [{K,V}]; none -> [] end "
				"end, ~w)", [ValsS, [ElementFun(KV) || KV <- KeyValues]]);
		(all) ->
			ValsS = io_lib:format("bbtree:from_orddict(~w)", [qlc_nodes(Tree, infinity, fun(X) -> X end)]),
			io_lib:format("bbtree:table(~s)", [ValsS])
	end,
	LookupFun = fun(1, Ks) ->
		lists:flatmap(fun(K) ->
			case lookup(K, Tree) of
				{value, V} -> [{K,V}];
				none -> []
			end
		end, Ks)
	end,
	Options = [
		{info_fun, InfoFun},
		{format_fun, FormatFun},
		{lookup_fun, LookupFun},
		{key_equality,'=='}],
	qlc:table(TraverseFun, Options).

table_reverse(Tree) ->
	table(Tree, [reverse]).

table(Tree) ->
	table(Tree, []).


qlc_next({K, V, Iterator}=NN) ->
% if we are using caching iterator, we should return whole cache to qlc
% (or 1 element less, to not fill new cache again, as it will double memory usage)
	[{K,V} | fun() -> qlc_next(next(Iterator)) end];
qlc_next(none) ->
	[].

qlc_nodes(Tree, infinity, ElementFun) ->
	qlc_nodes(Tree, -1, ElementFun);
qlc_nodes(Tree, NElements, ElementFun) ->
	qlc_iter(iterator(Tree), NElements, ElementFun).

qlc_iter(_Iterator, 0, _EFun) ->
	'...';
qlc_iter(Iterator1, N, EFun) ->
	case next(Iterator1) of
		{K, V, Iterator2} ->
			[EFun({K,V}) | qlc_iter(Iterator2, N-1, EFun)];
		none ->
			[]
	end.

% very simple iterator

% SubCurrent is a current index in SubTree
% (which will be equal, at some point, right subtree of original Tree,
%  and then right subtree of this right subtree, ...)
%  This accelerate iteration in late phase, and reduces memory usage,
%  which can be usefull if one converts one tree into something another.
%
next({Start, End, Current, SubCurrent, SubTree} = Iterator1) ->
	Inc = if Start =< End -> 1; true -> -1 end,
	if
		Current =/= End ->
			{K, V} = index_kv(Current, SubTree),
			Iterator2 = {Start, End, Current+Inc, SubTree},
			{K, V, Iterator2};
		true ->
			none
	end;
% for caching iterator i have few ideas for strategy
%   - constant sized cache: 30 seems good number
%   - tree dependant: sqrt(N) seems good number
%   - latency dependant: if interval beetwen call to this function is less than a execution time of this function, increase some factor (but with upper limit, of about 1000)
next({Start, End, Current, SubCurrent, SubTree, Cache} = Iterator1) ->
	Inc = todo,
	case cache_next(Cache) of
		{{K,V}, NewCache} ->
			Iterator2 = {Start, End, Current+Inc, SubTree, NewCache},
			{K, V, Iterator2};
		empty ->
			case none of
				none ->
					Iterator2 = {Start, End, Current+Inc, SubTree, []},
					{k, v, Iterator2};
				nn ->
					none
			end
	end.

% returns next Count elements
% this slighty amortizes some expensive traversal
next(Iterator1, Count) ->
	aaa.

iterator(Tree) ->
	iterator_range_i(1, size(Tree), Tree).

iterator_reverse(Tree) ->
	iterator_range_i(size(Tree), 1, Tree).

iterator_range_i(Start, End, Tree) when Start >= 0, End >= 0 ->
	Size = size(Tree),
	if
		(Start =< Size) and (End =< Size) ->
			Iterator = {Start, End, Start, 0, Tree},
			Iterator;
		true ->
			throw(badarg)
	end.

iterator_range_i_caching(Start, End, Tree) when Start >= 0, End >= 0 ->
	Size = size(Tree),
	if
		(Start =< Size) and (End =< Size) ->
			Iterator = {Start, End, Start, 0, Tree, cache_new()},
			Iterator;
		true ->
			throw(badarg)
	end.

cache_new() ->
	{[],[]}.

cache_next({[Next|Front], Rear}) ->
	NewCache = {Front, Rear},
	{Next, NewCache};
cache_next(_) ->
	empty.

