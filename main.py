# %% [markdown]
# # Final Project
# Ji Liu
# 
# This project utilizes a ***singly-linked list*** for the "Snake" game, written with Pygame. Unlike the conventional Snake game, the current game uses mouse to guide the snake, i.e., the head of the snake follows the mouse trajectory. In addition, some obstacles are added to the scene for difficulty.
# 
# ![gameplay](screenshot.png)
# 
# The compiled exe file for the game can be found at [***https://shorturl.at/7nHB6***](https://shorturl.at/7nHB6)
# 
# A video demo for the game can be found [***here***](https://www.dropbox.com/scl/fi/rrnvbeo5u1d9su0avmlhn/gameplay_vid.mp4?rlkey=ycb86jmroatbklz0jnd26s0ic&dl=0).
# 
# ## Data Structure
# 
# The basic unit of the snake is a "Node", which takes on different states (initialized, captured, collision). A Node is also the basic unit in the singly-linked list. The Snake instance stores the head of the singly-linked list. When the head of the Snake comes close to an uncaptured node (green), this node is inserted at the head of the list, i.e., it becomes the new head of the Snake. 
# 
# ## Movement
# 
# The head of the Snake is designed to follow the trajectory (exponentially smoothed) of the mouse cursor, while the body follows the past trajectory of the head. To achieve this function, the Snake class stores a ***buffer*** of the past coordinates of the head. In addition, it calculates the accumulative distance of these coordinates to the current location of the head along this trajectory. At each time step, the buffer append the latest values, as well as popping out values if the corresponding length is longer than the Snake length. 
# 
# The body segments' locations are updated as follows. First the length of the body segment is pre-defined. During update, the distance to the head, i.e., segment index times segment length, is used to query the accumulative distance buffer using ***binary search***, which returns the index through which we can interpolate the desired location along the stored past trajectory of the head. The head-body list is thus traversed in its entirety at each time step to determine the segment locations.
# 
# ## Code

# %%
import numpy as np
import pygame
from pygame.locals import (
    K_r,
    K_SPACE,
    K_ESCAPE,
    KEYDOWN,
    QUIT,
)
pygame.init()
myfont = pygame.font.SysFont("Arial", 30)

class Node(pygame.sprite.Sprite):
    """node of the snake

    Args:
        pygame (_type_): _description_

    Returns:
        _type_: _description_
    """
    COLOR_INIT = (34, 227, 147)
    COLOR_CAPTURED = (151, 23, 255)
    COLOR_COLLISION = (255, 0, 0)
    WHITE = (255,255,255)
    DIST = 0
    Kp = 0.1
    Kd = Kp/3.
    R=20

    def __init__(self, x=0, y=0,
                 coef=0.9):
        super().__init__()
        self.surf = pygame.Surface([Node.R*2, Node.R*2])
        self.surf.fill(Node.WHITE)
        self.surf.set_colorkey(Node.WHITE)
        pygame.draw.circle(surface=self.surf,
            color=self.COLOR_INIT,center=(Node.R,Node.R),
            radius=Node.R)
        
        self.rect = self.surf.get_rect()
        self.rect.x = x-Node.R
        self.rect.y = y-Node.R
        self.past_error = 0.0
        self.next = None
        self.buffer = []
        self.target_x = None
        self.target_y = None
        self.history_coef = coef

    @property
    def center(self):
        return self.rect.x+Node.R, self.rect.y+Node.R

    def update(self,x,y, head=True):
        if head:
            self._update_head(x,y)
        else:
            self._update(x,y)

    def set_color(self, c):
        pygame.draw.circle(surface=self.surf,
                           color=c,center=(Node.R,Node.R),
                            radius=Node.R)
        
    def _update_head(self, x, y):
        """the update logic for the head node,
        which follows the smoothed target point, i.e., mouse position

        Args:
            x (_type_): x position of the mouse
            y (_type_): y position of the mouse
        """
        cx,cy = self.center
        if self.target_x is None:
            self.target_x = x
            self.target_y = y
        else:
            self.target_x = self.history_coef*self.target_x + \
                (1-self.history_coef)*x
            self.target_y = self.history_coef*self.target_y + \
                (1-self.history_coef)*y
        x = self.target_x
        y = self.target_y
        
        d = np.sqrt((cx-x)**2 + (cy-y)**2) + 1e-6
        e = np.abs(d-self.DIST)
        de = e-self.past_error
        self.past_error = e
        e_ = (e*Node.Kp + de*Node.Kd)
        dx = (x-cx)/d*e_
        dy = (y-cy)/d*e_
        self.rect.x += dx
        self.rect.y += dy
    
    def _update(self, x, y):
        """update logic of non-head nodes, 
        simply set to the corresponding coord

        Args:
            x (_type_): _description_
            y (_type_): _description_
        """
        self.rect.x = x-Node.R
        self.rect.y = y-Node.R

def binary_search(arr, x, l=0):
    """binary search; can supply starting index with l

    Args:
        arr (_type_): _description_
        x (_type_): _description_
        l (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    l = l if l else 0
    r = len(arr)-1
    while l <= r:
        mid = l + (r-l)//2
        if arr[mid] == x:
            return mid
        elif arr[mid] < x:
            l = mid + 1
        else:
            r = mid - 1
    return l

class Snake():
    """the snake class that contains the head and the body, is represented using
    a singly linked list. The head is the first node and the body is the rest of
    the nodes. 

    Returns:
        _type_: _description_
    """
    COLLISION_SELF = 1
    COLLISION_WALL = 2
    COLLISION_OBS = 3

    def __init__(self,init_node,
                 width,height) -> None:
        self.body = pygame.sprite.Group()
        self.head = init_node
        self.head_traj = [list(init_node.center)]
        self.traj_cum_len = [0.0]
        
        self.seg_len = 2.1*Node.R # length of each segment
        self.capture_dist = self.seg_len
        self.n_nodes = 1
        self.canvas_width = width
        self.canvas_height = height

    def update(self,x,y):
        """update the snake. the trajectory of the head is stored in a buffer
        and the cumulative length of the trajectory is stored in traj_cum_len.
        each segment's position is calculated based on the interpolated coord,
        using the distance from the segment to the head. The interpolation is 
        calculated using binary search on the cumulative length of the traj.

        Args:
            x (_type_): x of the mouse position
            y (_type_): y of the mouse position
        """

        # update head trajectory
        cx,cy = self.head.center
        px,py = self.head_traj[0]
        self.head_traj.insert(0, [cx,cy])

        # update cumulative length
        d = np.sqrt((cx-px)**2 + (cy-py)**2)
        for i, _ in enumerate(self.traj_cum_len):
            self.traj_cum_len[i] += d
        self.traj_cum_len.insert(0, 0.0)

        # remove old trajectory if longer than needed
        while self.traj_cum_len[-1] > self.seg_len*(self.n_nodes+1):
            self.traj_cum_len.pop(-1)
            self.head_traj.pop(-1)
            if len(self.head_traj) == 0:
                break
        
        # update head
        self.head.update(x,y,head=True)

        # update body
        next = self.head.next
        ind = 1
        k_prev = None
        while next:
            next:Node
            sl = ind*self.seg_len
            # use binary search to find the pair of stored coordinate points
            # to interpolate the segment's position
            k = binary_search(self.traj_cum_len, sl, k_prev)
            if k<=0 or k>=len(self.traj_cum_len):
                break

            # store found index, so next time we can start from there, instead 
            # of starting at 0
            k_prev = k

            # do interpolation and found x, y
            rho = (sl-self.traj_cum_len[k-1]) / \
                (self.traj_cum_len[k]-self.traj_cum_len[k-1])
            x = (1-rho)*self.head_traj[k-1][0] + \
                rho*self.head_traj[k][0]
            y = (1-rho)*self.head_traj[k-1][1] + \
                rho*self.head_traj[k][1]
            
            # update body segment
            next.update(x,y,head=False)

            # move on to the next segment
            next = next.next
            ind+=1

    def capture(self, node : Node):
        """check if head is within some threshold of the food node
        node. if so, add the node to the snake as the new head, and move the old
        head to the body.

        Args:
            node (_type_): _description_
        """
        cx1,cy1 = self.head.center
        cx2,cy2 = node.center
        d = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
        if d < self.capture_dist:
            node.next = self.head
            # move prev head to body
            self.body.add(self.head)
            self.head = node
            node.set_color(Node.COLOR_CAPTURED)
            self.n_nodes += 1
            # print("captured")
            return 1
        return 0
    
    def check_collision(self, all_obs:pygame.sprite.Group):
        """check if head has collided with the body / wall / obstacle
        """
        if self._check_collision_self():
            return Snake.COLLISION_SELF
        if self._check_collision_wall():
            return Snake.COLLISION_WALL
        if self._check_collision_obs(all_obs):
            return Snake.COLLISION_OBS
        return 0

    def _check_collision_self(self):
        for node in self.body:
            cx1,cy1 = self.head.center
            cx2,cy2 = node.center
            d = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
            if d < 1.0*Node.R:
                self.head.set_color(Node.COLOR_COLLISION)
                node.set_color(Node.COLOR_COLLISION)
                return True
        return False

    def _check_collision_wall(self):
        cx,cy = self.head.center
        if cx < 0 or cx > self.canvas_width or \
            cy < 0 or cy > self.canvas_height:
            self.head.set_color(Node.COLOR_COLLISION)
            return True
        return False

    def _check_collision_obs(self, all_obs):
        for obs in all_obs:
            if pygame.sprite.collide_rect(self.head, obs):
                self.head.set_color(Node.COLOR_COLLISION)
                return True
        return False
    
class NodeSpawner:
    """node spawning class, that spawns a node at a random location that does 
    not collide with any other nodes or obstacles
    """
    def __init__(self,
                 all_nodes,
                 all_obs,
                 WIDTH, HEIGHT) -> None:
        self.all_nodes = all_nodes
        self.all_obs = all_obs
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT

    def spawn(self,x=None,y=None):
        if not x:
            redo=True
            while redo:
                x=np.random.randint(3*Node.R,self.WIDTH-3*Node.R)
                y=np.random.randint(3*Node.R,self.HEIGHT-3*Node.R)
                node = Node(x,y)
                if pygame.sprite.spritecollideany(node, self.all_nodes) or \
                    pygame.sprite.spritecollideany(node, self.all_obs):
                    del node
                    continue
                else:
                    redo=False
        else:
            x = np.clip(x, 3*Node.R, self.WIDTH-3*Node.R)
            y = np.clip(y, 3*Node.R, self.HEIGHT-3*Node.R)
        return Node(x,y)
    
class Obstacle(pygame.sprite.Sprite):
    """obstacle class that spawns a random rectangular obstacle at a random 
    location. overlapping with each other is fine

    Args:
        pygame (_type_): _description_
    """
    MAX_SIZE = 60
    MIN_SIZE = 20
    COLOR = (252, 3, 223)

    def __init__(self,canvas_width,canvas_height):
        super().__init__()
        self.width = np.random.randint(self.MIN_SIZE, self.MAX_SIZE)
        self.height = np.random.randint(self.MIN_SIZE, self.MAX_SIZE)
        self.surf = pygame.Surface([self.width, self.height])
        self.surf.fill(Obstacle.COLOR)
        self.rect = self.surf.get_rect()
        self.rect.x = np.random.randint(0, canvas_width-self.width)
        self.rect.y = np.random.randint(0, canvas_height-self.height)

    def update(self):
        pass


# %%
# Set up the drawing window
WIDTH = 500
HEIGHT = 500
n_obstacle = 3

screen = pygame.display.set_mode([WIDTH, HEIGHT])
clock = pygame.time.Clock()

def wait_for_start(screen, clock):
    """wait for the player to press space key to start the game

    Args:
        screen (_type_): _description_
        clock (_type_): _description_
    """
    screen.fill((255, 255, 255))
    text = myfont.render(f"Press space key to start", 1, (0,0,0))
    screen.blit(text, (0, 0))
    pygame.display.update()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == KEYDOWN:
                if event.key == K_SPACE:
                    running = False
            elif event.type == QUIT:
                running = False
        clock.tick(30)

def play(screen, clock):
    """handles the main game loop

    Args:
        screen (_type_): _description_
        clock (_type_): _description_
    """
    running = True

    # spawn obstacles
    all_obs = pygame.sprite.Group()
    all_nodes = pygame.sprite.Group()
    for _ in range(n_obstacle):
        obs = Obstacle(WIDTH, HEIGHT)
        all_obs.add(obs)

    # spawn the snake
    spawner = NodeSpawner(all_nodes,all_obs, WIDTH, HEIGHT)
    x,y = pygame.mouse.get_pos()
    init_node = spawner.spawn(x,y)
    init_node.set_color(Node.COLOR_CAPTURED)
    all_nodes.add(init_node)
    snake = Snake(init_node, WIDTH, HEIGHT)

    # spawn the first food node
    new_node = spawner.spawn()
    all_nodes.add(new_node)

    # main loop
    spawn_counter = 1
    capture_counter = 0
    game_over = False
    while running:
        # Look at every event in the queue
        for event in pygame.event.get():
            # Did the user hit a key?
            if event.type == KEYDOWN:
                # Was it the Escape key? If so  , stop the loop.
                if event.key == K_ESCAPE:
                    running = False

            # Did the user click the window close button? If so, stop the loop.
            elif event.type == QUIT:
                running = False

        if snake.capture(new_node):
            capture_counter += 1
        
        if capture_counter >= spawn_counter:
            new_node = spawner.spawn()
            all_nodes.add(new_node)
            spawn_counter += 1

        x, y = pygame.mouse.get_pos()
        snake.update(x,y)

        collision = snake.check_collision(all_obs)
        if collision > 0:
            running = False
            game_over = True
        if collision == Snake.COLLISION_SELF:
            print("collision with self, game over")
        elif collision == Snake.COLLISION_WALL:
            print("collision with wall, game over")       
        elif collision == Snake.COLLISION_OBS:
            print("collision with obstacle, game over")
            
        screen.fill((255, 255, 255))
        text = myfont.render(f"Score: {snake.n_nodes}", 1, (0,0,0))
        screen.blit(text, (0, 0))
        
        for obs in all_obs:
            screen.blit(obs.surf, obs.rect)

        for node in all_nodes:
            screen.blit(node.surf, node.rect)

        if game_over:
            text = myfont.render(f"Game Over", 1, (255,0,0))
            screen.blit(text, (WIDTH//2-50, HEIGHT//2-20))
        pygame.display.update()
        clock.tick(30)

def exit(screen, clock):
    """user chooses to exit or restart the game

    Args:
        screen (_type_): _description_
        clock (_type_): _description_

    Returns:
        _type_: _description_
    """
    screen.fill((255, 255, 255))
    text = myfont.render(f"Press R (reset) / Esc (exit)", 1, (0,0,0))
    screen.blit(text, (0, 0))
    pygame.display.update()
    running = True
    exit=False
    while running:
        for event in pygame.event.get():
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
                    exit=True
                elif event.key == K_r:
                    running = False
                    exit=False
            elif event.type == QUIT:
                running = False
                exit=True
        clock.tick(30)
    return exit

wait_for_start(screen, clock)
while True:
    play(screen, clock)
    pygame.time.wait(2000)
    if exit(screen, clock):
        break

pygame.quit()


