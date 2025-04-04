//+------------------------------------------------------------------+
//|                                               RL_Trader_EA.mq5    |
//|                                                                   |
//|                                                                   |
//+------------------------------------------------------------------+
#property copyright "RL Trader"
#property link      ""
#property version   "1.00"
#property strict

// Inclure la bibliothèque ZeroMQ
#include <Zmq/Zmq.mqh>

// Paramètres externes
input string ZmqHost = "127.0.0.1";
input int ZmqPubPort = 5555;  // Port pour publier les données vers Python
input int ZmqSubPort = 5556;  // Port pour recevoir les commandes de Python
input int DataSendInterval = 5; // Secondes

// Variables globales
Context context;
Socket pubSocket(context, ZMQ_PUB);
Socket subSocket(context, ZMQ_SUB);
datetime lastSendTime = 0;
int timerInterval = 1000; // 1 seconde en millisecondes

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   // Afficher des informations de débogage
   Print("OnInit started");
   
   // Initialiser les sockets ZeroMQ
   Print("Initializing ZeroMQ sockets...");
   
   // Essayer d'initialiser le socket de publication
   Print("Binding pub socket...");
   bool bind_result;
   string bind_address = "tcp://" + ZmqHost + ":" + IntegerToString(ZmqPubPort);
   bind_result = pubSocket.bind(bind_address);
   Print("Pub socket bind result: ", bind_result ? "success" : "failed");
   
   if(!bind_result)
   {
      Print("Failed to bind pub socket to ", bind_address);
      return INIT_FAILED;
   }
   
   // Essayer d'initialiser le socket d'abonnement
   Print("Connecting sub socket...");
   bool connect_result;
   string connect_address = "tcp://" + ZmqHost + ":" + IntegerToString(ZmqSubPort);
   connect_result = subSocket.connect(connect_address);
   Print("Sub socket connect result: ", connect_result ? "success" : "failed");
   
   if(!connect_result)
   {
      Print("Failed to connect sub socket to ", connect_address);
      return INIT_FAILED;
   }
   
   subSocket.subscribe("");
   
   // Configurer le timer
   EventSetTimer(1);
   
   Print("RL Trader EA initialized. Publishing on port ", ZmqPubPort, ", subscribing on port ", ZmqSubPort);
   
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   // Arrêter le timer
   EventKillTimer();
   
   // Fermer les sockets
   pubSocket.unbind("tcp://" + ZmqHost + ":" + IntegerToString(ZmqPubPort));
   subSocket.disconnect("tcp://" + ZmqHost + ":" + IntegerToString(ZmqSubPort));
   
   Print("RL Trader EA deinitialized");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // Vérifier les commandes entrantes
   CheckForCommands();
}

//+------------------------------------------------------------------+
//| Timer function                                                   |
//+------------------------------------------------------------------+
void OnTimer()
{
   // Envoyer les données de marché à intervalle régulier
   if(TimeCurrent() - lastSendTime >= DataSendInterval)
   {
      SendMarketData();
      lastSendTime = TimeCurrent();
   }
}

//+------------------------------------------------------------------+
//| Trade function                                                   |
//+------------------------------------------------------------------+
void OnTrade()
{
   // Envoyer les mises à jour des positions lorsqu'un trade est exécuté
   SendPositionUpdates();
}

//+------------------------------------------------------------------+
//| Envoyer les données de marché à Python                           |
//+------------------------------------------------------------------+
void SendMarketData()
{
   string message = "";
   
   // Ajouter les données OHLCV actuelles
   message += "MARKET_DATA|";
   message += Symbol() + "|";
   message += DoubleToString(SymbolInfoDouble(Symbol(), SYMBOL_BID)) + "|";
   message += DoubleToString(SymbolInfoDouble(Symbol(), SYMBOL_ASK)) + "|";
   
   // Ajouter les données des dernières bougies
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   int copied = CopyRates(Symbol(), PERIOD_M15, 0, 5, rates);
   
   if(copied > 0)
   {
      for(int i = 0; i < copied; i++)
      {
         message += DoubleToString(rates[i].open) + "|";
         message += DoubleToString(rates[i].high) + "|";
         message += DoubleToString(rates[i].low) + "|";
         message += DoubleToString(rates[i].close) + "|";
         message += IntegerToString(rates[i].tick_volume) + "|";
      }
   }
   
   // Ajouter des informations sur le compte
   message += DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE)) + "|";
   message += DoubleToString(AccountInfoDouble(ACCOUNT_EQUITY)) + "|";
   message += IntegerToString(PositionsTotal());
   
   // Envoyer le message
   ZmqMsg request(message);
   pubSocket.send(request);
   
   Print("Market data sent: ", message);
}

//+------------------------------------------------------------------+
//| Envoyer les mises à jour des positions                           |
//+------------------------------------------------------------------+
void SendPositionUpdates()
{
   string message = "POSITION_UPDATE|";
   
   // Ajouter des informations sur les positions ouvertes
   int total = PositionsTotal();
   message += IntegerToString(total) + "|";
   
   for(int i = 0; i < total; i++)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket <= 0) continue;
      
      if(PositionSelectByTicket(ticket))
      {
         message += IntegerToString(ticket) + "|";
         message += PositionGetString(POSITION_SYMBOL) + "|";
         message += DoubleToString(PositionGetDouble(POSITION_VOLUME)) + "|";
         message += DoubleToString(PositionGetDouble(POSITION_PRICE_OPEN)) + "|";
         message += DoubleToString(PositionGetDouble(POSITION_SL)) + "|";
         message += DoubleToString(PositionGetDouble(POSITION_TP)) + "|";
         message += DoubleToString(PositionGetDouble(POSITION_PROFIT)) + "|";
         message += IntegerToString(PositionGetInteger(POSITION_TYPE)) + "|";
      }
   }
   
   // Envoyer le message
   ZmqMsg request(message);
   pubSocket.send(request);
   
   Print("Position update sent: ", message);
}

//+------------------------------------------------------------------+
//| Vérifier les commandes entrantes de Python                       |
//+------------------------------------------------------------------+
void CheckForCommands()
{
   // Essayer de recevoir un message sans polling
   ZmqMsg message;
   
   // Utiliser recv avec ZMQ_DONTWAIT pour éviter le blocage
   if(subSocket.recv(message, ZMQ_DONTWAIT))
   {
      // Traiter le message
      string content = message.getData();
      ProcessCommand(content);
   }
}

//+------------------------------------------------------------------+
//| Traiter une commande reçue                                       |
//+------------------------------------------------------------------+
void ProcessCommand(string command)
{
   // Diviser la commande en parties
   string parts[];
   StringSplit(command, '|', parts);
   
   if(ArraySize(parts) < 2) return;
   
   Print("Received command: ", command);
   
   // Traiter selon le type de commande
   if(parts[0] == "OPEN_ORDER")
   {
      // Format: OPEN_ORDER|BUY/SELL|VOLUME|SL|TP
      if(ArraySize(parts) >= 5)
      {
         string type = parts[1];
         double volume = StringToDouble(parts[2]);
         double sl = StringToDouble(parts[3]);
         double tp = StringToDouble(parts[4]);
         
         MqlTradeRequest request = {};
         MqlTradeResult result = {};
         
         request.action = TRADE_ACTION_DEAL;
         request.symbol = Symbol();
         request.volume = volume;
         request.sl = sl;
         request.tp = tp;
         request.deviation = 10;
         request.type_filling = ORDER_FILLING_FOK;
         request.comment = "RL Trader";
         
         if(type == "BUY")
         {
            request.type = ORDER_TYPE_BUY;
            request.price = SymbolInfoDouble(Symbol(), SYMBOL_ASK);
         }
         else if(type == "SELL")
         {
            request.type = ORDER_TYPE_SELL;
            request.price = SymbolInfoDouble(Symbol(), SYMBOL_BID);
         }
         
         bool sent = OrderSend(request, result);
         
         if(sent)
         {
            Print("Order sent successfully. Result code: ", result.retcode);
         }
         else
         {
            Print("Failed to send order. Error: ", GetLastError());
         }
      }
   }
   else if(parts[0] == "CLOSE_ORDER")
   {
      // Format: CLOSE_ORDER|TICKET
      if(ArraySize(parts) >= 2)
      {
         ulong ticket = StringToInteger(parts[1]);
         
         if(PositionSelectByTicket(ticket))
         {
            MqlTradeRequest request = {};
            MqlTradeResult result = {};
            
            request.action = TRADE_ACTION_DEAL;
            request.position = ticket;
            request.symbol = PositionGetString(POSITION_SYMBOL);
            request.volume = PositionGetDouble(POSITION_VOLUME);
            request.deviation = 10;
            request.comment = "RL Trader Close";
            
            if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
            {
               request.type = ORDER_TYPE_SELL;
               request.price = SymbolInfoDouble(request.symbol, SYMBOL_BID);
            }
            else
            {
               request.type = ORDER_TYPE_BUY;
               request.price = SymbolInfoDouble(request.symbol, SYMBOL_ASK);
            }
            
            bool sent = OrderSend(request, result);
            
            if(sent)
            {
               Print("Position closed successfully. Result code: ", result.retcode);
            }
            else
            {
               Print("Failed to close position. Error: ", GetLastError());
            }
         }
      }
   }
   else if(parts[0] == "CLOSE_ALL")
   {
      CloseAllPositions();
   }
}

//+------------------------------------------------------------------+
//| Fermer toutes les positions ouvertes                             |
//+------------------------------------------------------------------+
void CloseAllPositions()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket <= 0) continue;
      
      if(PositionSelectByTicket(ticket))
      {
         MqlTradeRequest request = {};
         MqlTradeResult result = {};
         
         request.action = TRADE_ACTION_DEAL;
         request.position = ticket;
         request.symbol = PositionGetString(POSITION_SYMBOL);
         request.volume = PositionGetDouble(POSITION_VOLUME);
         request.deviation = 10;
         request.comment = "RL Trader Close All";
         
         if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
         {
            request.type = ORDER_TYPE_SELL;
            request.price = SymbolInfoDouble(request.symbol, SYMBOL_BID);
         }
         else
         {
            request.type = ORDER_TYPE_BUY;
            request.price = SymbolInfoDouble(request.symbol, SYMBOL_ASK);
         }
         
         bool sent = OrderSend(request, result);
         
         if(sent)
         {
            Print("Position closed successfully. Result code: ", result.retcode);
         }
         else
         {
            Print("Failed to close position. Error: ", GetLastError());
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Tester function                                                  |
//+------------------------------------------------------------------+
double OnTester()
{
   double ret = 0.0;
   // Calculer le ratio de Sharpe pour l'optimisation
   double profit = TesterStatistics(STAT_PROFIT);
   double max_drawdown = TesterStatistics(STAT_BALANCE_DD);
   double trades = TesterStatistics(STAT_TRADES);
   
   if(max_drawdown > 0 && trades > 10)
   {
      ret = profit / max_drawdown;
   }
   
   return ret;
}