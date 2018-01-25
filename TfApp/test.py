#import tensorflow as tf
#w = tf.Variable(1.0)
#ema = tf.train.ExponentialMovingAverage(0.9)
#update = tf.assign_add(w, 1.0)
#
#ema_op = ema.apply([update])
#with tf.control_dependencies([ema_op]):
#    ema_val = tf.identity(ema.average(update))
#
#sess = tf.Session()
#sess.run(tf.global_variables_initializer())
#for i in range(3):
#    print(sess.run([ema_val]))
#sess.close()
#
#import tensorflow as tf
#import cv2
#from tensorflow.python.client import device_lib
#
#def get_available_gpus():
#    local_device_protos = device_lib.list_local_devices()
#    return [x.name for x in local_device_protos if x.device_type == 'GPU']
#
#
#print(len(get_available_gpus()))


#def get_no_of_instances(cls_obj):
#    return cls_obj.no_inst
#class Kls(object):
#    no_inst = 0
#    def __init__(self):
#        Kls.no_inst = Kls.no_inst + 1
#ik1 = Kls()
#ik2 = Kls()
#print(get_no_of_instances(Kls))


#def iget_no_of_instance(ins_obj):
#    return ins_obj.__class__.no_inst
#class Kls(object):
#    no_inst = 0
#    
#    def __init__(self):
#        Kls.no_inst = Kls.no_inst + 1
#        print(Kls.no_inst)
#
#ik1 = Kls()
#ik2 = Kls()
##print(iget_no_of_instance(ik1))

#class Date(object):
#
#    def __init__(self, day=0, month=0, year=0):
#        self.day = day
#        self.month = month
#        self.year = year
#
#    @classmethod
#    def from_string(cls, date_as_string):
#        day, month, year = map(int, date_as_string.split('-'))
#        date1 = cls(day, month, year)
#        return date1
#
#    @staticmethod
#    def is_date_valid(date_as_string):
#        day, month, year = map(int, date_as_string.split('-'))
#        Date.day=day
#        return day <= 31 and month <= 12 and year <= 3999
#
#
#date2 = Date.from_string('11-09-2012')
#print(date2.day)
#
#a = Date()
#print(a.day)
#
#is_date = Date.is_date_valid('11-09-2012')
#
#print(is_date)
#print(Date.day)

depth =5 
bn = [True] * 4

bn.append(False)

print(bn)


