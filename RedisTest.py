import redis
import numpy

r = redis.Redis(host='127.0.0.1', port=6379, db=0, decode_responses=True)

p = r.pubsub()
p.psubscribe("update")

size = 9
resolution = .5


def main():
    
    for update in p.listen():
        shootMode = r.get("mode")
        swapTime = r.get("swapTime")
        gunners = parsePositions("gunners")
        chargers = parsePositions("chargers")
        rocks = parsePositions("rocks")
        holes = parsePositions("holes")
        r.set("ready?", "yes")
        if r.get("quit") == "yes":
            r.set("quit","no")
            break
    
        
def parsePositions(redisKey):
    temp = r.lpop("gunners")
    result = numpy.zeros((size, size))
    while temp != None:
        temp = int(temp)
        print(temp)
        result[temp // size,temp % size] = 1
        temp = r.lpop("gunners")
    return result 


main()