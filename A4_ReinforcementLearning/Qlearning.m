%% Initialization
%  Initialize the world, Q-table, and hyperparameters

%What world to explore
world = 4;

%Initialize world
s = gwinit(world);
gwdraw()

%Actions to take, represents down up right and left
actions = [1,2,3,4];

%Probability
prob_a = ones(size(actions)) / size(actions,2);

%Initializes the Q matrix
Q = zeros(s.ysize,s.xsize, size(actions,2));

%Parameters
eta = 0.3; % Learning rate
gamma = 0.9; % Discount factor
epsilon = 0.9; % Exploration factor

%Nr of episodes
episodes = 1000;

%Greedy exploration till epsilon reaches 0.1
epsilon_update = (epsilon - 0.1)/episodes;


%% Training loop
%  Train the agent using the Q-learning algorithm.

for episode=1:episodes
    % do something
    s = gwinit(world);
    
    while s.isterminal == 0
        [action, opt_action] = chooseaction(Q, s.pos(1), s.pos(2), actions, prob_a, epsilon);
        s_next = gwaction(action);
        
        %Check if action is wanted
        %diff_pos = [(action==1) - (action==2); (action==3) - (action==4)];
        %next_pos = s.pos + diff_pos;
        
        %if next_pos ~= s_next.pos
        %    s = s_next;
        %    continue;
        %end
        
        if s_next.isvalid == 1
            r = s_next.feedback;
            V = getvalue(Q);
            gwstate()
            Q(s.pos(1), s.pos(2), action)  = (1-eta)*Q(s.pos(1), s.pos(2), action) + eta*(r + gamma*V(s_next.pos(1), s_next.pos(2)));
            s = s_next;
        else %Penalty
            Q(s.pos(1), s.pos(2), action) = -inf;
        end
    end
    
    epsilon = epsilon - epsilon_update;
    episode
end
%%
figure(1)
imagesc(Q(:,:,1))

figure(2)
imagesc(Q(:,:,2))

figure(3)
imagesc(Q(:,:,3))

figure(4)
imagesc(Q(:,:,4))


%% Test loop
%  Test the agent (subjectively) by letting it use the optimal policy
%  to traverse the gridworld. Do not update the Q-table when testing.
%  Also, you should not explore when testing, i.e. epsilon=0; always pick
%  the optimal action.
s = gwinit(world);

n_actions = 0;

P = getpolicy(Q);
V = getvalue(Q);

figure(1)
gwdraw()
gwdrawpolicy(P)

figure(2)
imagesc(V)

while s.isterminal == 0
    action = P(s.pos(1), s.pos(2));
    
    s_next = gwaction(action);
    s = s_next;
    n_actions = n_actions + 1;
end

