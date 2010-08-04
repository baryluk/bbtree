-module(bbtree_wrap, [BBTree0]).
-author('baryluk@smp.if.uj.edu.pl').

-export([
	clone/0,             % empty() -> BB.                                                      % from gb_trees
	is_bb/0,             % is_bb() -> true.
	is_empty/0,          % is_empty() -> true | false.                                       % from dict and gb_trees
	size/0,              % size() -> integer().                                              % from dict and gb_trees
	is_key/1,            % is_key(Key) -> true | false.                                    % from dict
	is_defined/1,        % same as is_key/1                                                    % from gb_trees and proplists
	is_member/1,         % same as is_key/1                                                    % from gb_sets
	is_element/1,        % same as is_key/1                                                    % from gb_sets
	store/2,             % store(Key, Value) -> BB2.                                       % from dict
	enter/2,             % same as store/2                                                     % from gb_trees
	insert/2,            % insert(Key, Value) -> BB2 | crash.                              % from gb_trees
%	save/2,              % save(Key, Value) -> {none | {ok, OldValue}, BB2}.
	erase/1,             % erase(Key) -> BB2 | crash.                                      % from dict
	delete/1,            % same as erase/1                                                     % from gb_trees
	delete_any/1,        % delete(Key) -> BB2.                                             % from gb_trees
%	take/1,              % take(Key,) -> {OldValue, BB2} | crash.
%	take/2,              % take(Key, Default) -> {OldValue | Default, BB2}.
	largest/0,           % largest() -> {K, V} | crash.                                      % from gb_trees
	smallest/0,          %                                                                     % from gb_trees
	take_largest/0,      % take_largest() -> {K, V, BB2} | crash.                            % from_gbrees
	take_smallest/0,     %                                                                     % from_gbrees
	fold/2,              % fold(fun(K,V,Acc1) -> Acc2 end, Acc0)   -> Acc3.                % from dict
	foldl/2, foldr/2,    % foldl(fun(K,V,Acc1) -> Acc2 end, Acc0)   -> Acc3.
	foldl2/2, foldr2/2,  % foldl2(fun({K,V},Acc1) -> Acc2 end, Acc0) -> Acc3.
	update/2,            % update(Key, fun(OldV) -> NewV end) -> BB2 | crash.              % from dict
	update/3,            % update(Key, fun(OldV) -> NewV end, InitV) -> BB2.               % from dict
	update_lazy/3,       % update_lazy(Key, fun(OldV) -> NewV end, fun() -> InitV end, BB) -> BB2.
	update_counter/2,    % update_counter(Key, Inc, BB) -> BB2.                                % from dict
	update_counter/3,    % update_counter(Key, Inc, Init, BB) -> BB2.
	to_list/0,           % to_list(BB) -> [{K1,V1},{K2,V2},...].                               % from dict and gb_trees
	fetch/1,             % fetch(Key, Dict) -> V | crash.                                      % from dict
	fetch/2,             % fetch(Key, Dict, Default) -> V | Default.
	fetch_keys/0,        % fetch_keys(BB) -> [K1,K2,...].                                      % from dict
	keys/0,              % keys(BB) -> [K1,K2,...].                                            % from gb_trees
	keys_r/0,            % same as keys/0, but list is reversed
	get_keys/0,          % same as keys/0                                                      % from proplists
	find/1,              % find(Key, BB) -> {ok, V} | error.                                   % from dict
	find/2,              % find(Key, BB, Default) -> {ok, V} | {default, Default}.
	get_value/1,         % get_value(Key) -> V | undefined.                                % from proplists
	get_value/2,         % get_value(Key, Default) -> V | Default.                         % from proplists
	lookup/1,            % lookup(Key) -> {value, V} | none.                               % from gb_trees
	                     %  in proplists there is similary lookup(K,List) -> tuple() | none.
	get/1,               % get(Key) -> Value | crash.                                      % from gb_trees
	foreachl/1,          % foreachl(fun(K,V) -> xx end) -> void().
	foreach/1,           % same as foreachl/1, similar to lists:foreach
	map/1,               % map(fun(K, V1) -> V2 end) -> BB2.                               % from dict and gb_trees
	map_foldl/2,         % map_foldl(fun(K, V1, Acc) -> {V2, Acc2} end, Acc0) -> {BB2, Acc3}.
	map_foldr/2,         % map_foldr(fun(K, V1, Acc) -> {V2, Acc2} end, Acc0) -> {BB2, Acc3}.
	map_reduce/3,        % map_reduce(fun(K, V1) -> [X] end, fun(X1, X2) -> X3 end, X0) -> X4.
	values/0,            % values() -> [V1,V2,...]                                           % from gb_trees
	values_r/0,          % same as values/0, but list is reversed
%	iterator/0,          % iterator() -> iterator().                                         % from gb_trees
%	filter/1,            % filter(fun(Key, Value) -> true | false end) -> BB2.             % from dict
	rank_kv/1,           % rank_kv(N) -> {Key,Value}
	kv_rank/1,           % kv_rank(Key) -> {N,Value}
%	count_kv/2,          % count_kv(Key1, Key2) -> integer().
%	subtree_rank/2,      % count_kv(I, J) -> BB2.
%	subtree_kv/2         % count_kv(Key1, Key2) -> BB2.
	table/0,             % for QLC
	unwrap/0,            % unwrap parametrized module back to normal tree
	wrap/0,               % return same module again
	stat/0,
	is_correct/0
	]).

% recomenened api:
%
%   empty/0,
%   is_defined/2,
%   save/3,
%
%

-define(MASTER, bbtree).

% rewrap
-define(BBTree, {bbtree_wrap,BBTree0}).

% Time complexity: O(1)
clone() ->
	?MODULE:new(BBTree0).

% Time complexity: O(1)
size() ->
	?MASTER:size(?BBTree).


% Time complexity: O(1)
is_bb() ->
	?MASTER:is_bb(?BBTree).

% Time complexity: O(1)
is_empty() ->
	?MASTER:is_empty(?BBTree).

% Time complexity: O(log n)
is_key(Key) ->
	?MASTER:is_key(Key, ?BBTree).

% Time complexity: O(log n)
is_defined(Key) ->
	?MASTER:is_defined(Key, ?BBTree).

% Time complexity: O(log n)
is_element(Key) ->
	?MASTER:is_key(Key, ?BBTree).

% Time complexity: O(log n)
is_member(Key) ->
	?MASTER:is_member(Key, ?BBTree).

% Time complexity: O(log n)
find(Key) ->
	?MASTER:find(Key, ?BBTree).

% Time complexity: O(log n)
find(Key, Default) ->
	?MASTER:find(Key, Default, ?BBTree).

% Time complexity: O(log n)
fetch(Key) ->
	?MASTER:fetch(Key, ?BBTree).

% Time complexity: O(log n)
fetch(Key, Default) ->
	?MASTER:fetch(Key, Default, ?BBTree).

% Time complexity: O(log n)
lookup(Key) ->
	?MASTER:lookup(Key, ?BBTree).

% Time complexity: O(log n)
get(Key) ->
	?MASTER:get(Key, ?BBTree).

% Time complexity: O(log n)
get_value(Key) ->
	?MASTER:get_value(Key, ?BBTree).

% Time complexity: O(log n)
get_value(Key, Default) ->
	?MASTER:get_value(Key, ?BBTree, Default).

%%%% SIMPLE ITERATING

% Time complexity: O(n)
foldl(Fun, Acc0) ->
	?MASTER:foldl(Acc0, Fun, ?BBTree).

% Time complexity: O(n)
foldr(Fun, Acc0) ->
	?MASTER:foldr(Acc0, Fun, ?BBTree).

% Time complexity: O(n)
foldl2(Fun, Acc0) ->
	?MASTER:foldl2(Acc0, Fun, ?BBTree).

% Time complexity: O(n)
foldr2(Fun, Acc0) ->
	?MASTER:foldr2(Acc0, Fun, ?BBTree).

% Time complexity: O(n)
foreachl(Fun) ->
	?MASTER:foreachl(Fun, ?BBTree).

% Time complexity: O(n)
foreach(Fun) ->
	?MASTER:foreach(Fun, ?BBTree).


% Time complexity: O(n)
fold(Fun, Acc0) ->
	?MASTER:fold(Fun, Acc0, ?BBTree).


%%%% EXPORTING

% Time complexity: O(n)
to_list() ->
	?MASTER:to_list(?BBTree).

% Time complexity: O(n)
keys() ->
	?MASTER:keys(?BBTree).

% Time complexity: O(n)
keys_r() ->
	?MASTER:keys_r(?BBTree).

% Time complexity: O(n)
fetch_keys() ->
	?MASTER:fetch_keys(?BBTree).

% Time complexity: O(n)
get_keys() ->
	?MASTER:get_keys(?BBTree).

% Time complexity: O(n)
values() ->
	?MASTER:values(?BBTree).

% Time complexity: O(n)
values_r() ->
	?MASTER:values_r(?BBTree).


%%% MAPS  (Advanced iterations)

% Time complexity: O(n)
map(Fun) ->
	?MASTER:map(Fun, ?BBTree).

% Time complexity: O(n)
map_foldl(Fun, Acc0) ->
	{T, A} = ?MASTER:map_foldl(Fun, Acc0, ?BBTree),
	{T, A}.

% Time complexity: O(n)
map_foldr(Fun, Acc0) ->
	{T, A} = ?MASTER:map_foldr(Fun, Acc0, ?BBTree),
	{T, A}.

% Time complexity: O(n)
map_reduce(FunMap, FunReduce, Reduce0) ->
	?MASTER:map_reduce(FunMap, FunReduce, Reduce0, ?BBTree).

% external API for addintion
% Time complexity: O(log n)
store(Key, Value) ->
	?MASTER:store(Key, Value, ?BBTree).

% Time complexity: O(log n)
enter(Key, Value) ->
	?MASTER:enter(Key, Value, ?BBTree).

% Time complexity: O(log n)
insert(Key, Value) ->
	?MASTER:insert(Key, Value, ?BBTree).


% external API for deletion
% Time complexity: O(log n)
erase(Key) ->
	?MASTER:erase(Key, ?BBTree).

delete(Key) ->
	?MASTER:delete(Key, ?BBTree).

delete_any(Key) ->
	?MASTER:delete_any(Key, ?BBTree).

% Time complexity: O(log n)
smallest() ->
	?MASTER:smallest(?BBTree).

% Time complexity: O(log n)
largest() ->
	?MASTER:largest(?BBTree).

% Time complexity: O(log n)
take_smallest() ->
	?MASTER:take_smallest(?BBTree).

% Time complexity: O(log n)
take_largest() ->
	?MASTER:take_largest(?BBTree).



%%%% UPDATES

update_lazy(Key, Fun, InitFun) ->
	?MASTER:update_lazy(Key, Fun, InitFun, ?BBTree).

% this 4 functions bellow should be specialized, so no unacassary funs are created.

update(Key, Fun) ->
	?MASTER:update(Key, Fun, ?BBTree).

update(Key, Fun, Init) ->
	?MASTER:update(Key, Fun, Init, ?BBTree).

update_counter(Key, Inc) ->
	?MASTER:update_counter(Key, Inc, ?BBTree).

update_counter(Key, Inc, InitV)  ->
	?MASTER:update_counter(Key, Inc, InitV, ?BBTree).



% finds {K,V}, which is N'th in the Tree, if all K would be sorted
rank_kv(N) ->
	?MASTER:rank_kv(N, ?BBTree).

% tells what is the rank of K
kv_rank(K) ->
	?MASTER:kv_rank(K, ?BBTree).

% for QLC
table() ->
	?MASTER:table(?BBTree).


unwrap() ->
	?BBTree.

wrap() ->
	?MODULE:new(BBTree0).

is_correct() ->
	?MASTER:is_correct(?BBTree).

stat() ->
	?MASTER:stat(?BBTree).
