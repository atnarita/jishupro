void setup()
{
  Serial.begin(9600);
  Serial.println("start");
}

void loop()
{
  long x, y, z;
  x = analogRead(3);
  y = analogRead(4);
  z = analogRead(5);
//  String acc_sen = ;
  Serial.println((String)x+" "+y+" "+z);
//  Serial.print(" ");
//  Serial.print(y);
//  Serial.print(" ");
//  Serial.println(z);
  delay(100);
}
